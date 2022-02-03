"""Functions and utilities for working with experiment configurations."""

import configargparse
import datetime
import multiprocessing
import numpy as np
import os
import yaml

import util


# TODO: Functionality to add for parity with OptimisingLRSchedule
#   RBFs and length scales for 3F8
#   Parallel sequences etc. for PTB
#   Training grad clipping (for PTB)


def convert_scientific_notation(config):
    """Yaml automatically converts floats to scientific notation strings
    where appropriate. Unfortunately, neither the to.tensor nor the optimiser
    constructors seem to know what to do with this, since they're strings
    and not floats.
    This method converts fields known to be floats into floats if needed.
    We may want to try converting all fields instead, if we have many
    such fields.
    """
    keys_to_convert = ('lr', 'curvature_transform', 'weight_decay')
    for key in keys_to_convert:
        if key in config and isinstance(config[key], str):
            config[key] = float(config[key])
        elif (key == 'curvature_transform'
              and key in config
              and isinstance(config[key]['diagonal_value'], str)):
            config[key]['diagonal_value'] = float(config[key]['diagonal_value'])


def get_args(config):
    """Remove any 'class' entry from `config`, leaving the arguments to be
    supplied to the corresponding constructor function.
    """
    # Shallow copy, so references to PyTorch objects remain intact
    kwargs = config.copy()
    kwargs.pop('class', None)
    return kwargs


class ConfigFileAction(configargparse.Action):
    """Argparse action to process a config file from its path, allowing
    multiple files to be specified.
    """

    def __call__(self, parser, namespace, values, option_string=None):
        for config_file in values:
            with open(config_file, 'r') as config_text:
                config_updates = yaml.safe_load(config_text)
            # Can't treat a namespace as a dictionary, so deal with the first
            # level of nesting ourselves
            for key, new_value in config_updates.items():
                if isinstance(new_value, dict) and key in namespace:
                    util.nested_update(getattr(namespace, key), new_value)
                else:
                    setattr(namespace, key, new_value)


def load_config():
    """Parse command line arguments and config files, making
    interpretations/adjustments as needed.
    """
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser)

    parser.add('-c', '--config',
               action=ConfigFileAction,
               nargs='*',
               help='config file path')
    parser.add('--config_dicts',
               type=yaml.safe_load,
               help="Yaml specifying parameters for the experiment (dataset,"
               " model, loss, etc). See example configs",
               default={})
    parser.add('-d', '--device',
               choices=['cuda', 'cpu'],
               help='Device (cpu or cuda)')
    parser.add('--batch_size',
               type=int,
               help='Size of the batch to use in training')
    parser.add('--network_weight_steps',
               type=int,
               help='How many network weight update steps to do')
    parser.add('--hyperparameter_steps',
               type=int,
               help='How many hyperparameter update steps to do')
    parser.add('--total_step_limit',
               type=int,
               default=None,
               help='Upper bound on total number of steps (supports non-multiples of individual step numbers)')
    parser.add('--validation_proportion',
               type=float,
               help='Proportion of the training set to use for validation',
               default=0)
    parser.add('--optimised_hyperparameters',
               help='List of hyperparameters to be optimised',
               nargs='+',
               default=[])
    parser.add('--high_dimensional_hyperparameters',
               help='List of hyperparameters to expand per-parameter',
               nargs='+',
               default=[])
    parser.add('--network_weight_grad_clipping',
               type=yaml.safe_load,
               help='Whether or not (and how) to clip network_weight gradients',
               default={})
    parser.add('--hyperparameter_clipping',
               type=yaml.safe_load,
               help='Whether or not to clip hyperparameters',
               default={})
    parser.add('--transformed_hyperparameters',
               type=yaml.safe_load,
               help='Whether or not to transform hyperparameters',
               default={})
    parser.add('--reset_model_after_hyperparameter_update',
               type=yaml.safe_load,
               help='Whether to reset the model to its original state after updating hyperparameters')
    parser.add('--reset_loop_before_hyperparameter_step',
               type=int,
               default=-1,
               help='Restrict lookback distance by resetting gradient tracking this many steps before a hyperparameter update step')
    parser.add('--renew_model_reset_state_interval',
               type=int,
               default=None,
               help='Interval at which to update the model\'s reset state with the current state')
    parser.add('--patch_optimiser',
               type=yaml.safe_load,
               help='Whether to monkey-patch the network_weight optimiser to be differentiable using Higher')
    parser.add('--null_zero_validation_datasets',
               type=yaml.safe_load,
               default=True,
               help='Whether to use null validation datasets when validation_proportion is 0')
    parser.add('--force_single_network_weight_step',
               type=yaml.safe_load,
               default=False,
               help='Whether to force network_weight_steps=1, adjusting hyperparameter_steps accordingly')
    parser.add('--multi_batch_test_dataset_evaluations',
               type=yaml.safe_load,
               default=False,
               help='Whether to force test loss tracking to use all test batches in its computation')
    parser.add('--full_batch_validation_evaluations',
               type=yaml.safe_load,
               default=False,
               help='Whether to force validation loss computation to use full-dataset batches')
    parser.add('--penn_treebank_validation_override',
               type=yaml.safe_load,
               default=False,
               help='Override to support full_batch_validation_evaluations with PennTreebank')
    parser.add('-S', '--save_state',
               type=str,
               default=None,
               help='File in which to save final training state')
    parser.add('-s', '--load_state',
               type=str,
               default=None,
               help='File from which to load initial training state')
    parser.add('-l', '--log_root',
               type=str,
               help='Root directory for logging run data',
               default='./runs')
    parser.add('-g', '--run_group_name',
               type=str,
               help='run group name',
               default='Untitled')
    parser.add('-n', '--run_name',
               type=str,
               help='run name')
    args = parser.parse_args()

    config_dict = dict(args._get_kwargs())
    config_dict.pop('config')

    for key in ('network_weight', 'hyperparameter'):
        if key in config_dict['config_dicts']:
            convert_scientific_notation(config_dict['config_dicts'][key])

    return config_dict


def log_directory(config_dict):
    """Compute, create and return the appropriate log directory, removing
    consumed components of the configuration.
    """
    # Temporarily include extra random number for robustness
    run_prefix = datetime.datetime.now().isoformat() + '-' + str(np.random.rand())
    run_suffix = config_dict.pop('run_name')
    log_root = config_dict.pop('log_root')
    if run_suffix is None:
        run_name = run_prefix
    else:
        run_name = ' '.join((run_prefix, run_suffix))
    return os.path.join(log_root, config_dict.pop('run_group_name'), run_name)


def parallel_process_id():
    """Return the 0-based ID of the current parallel process.
    """
    process_name = multiprocessing.current_process().name
    if process_name.startswith('ForkPoolWorker-'):
        return int(process_name[15:])
    else:
        return 0

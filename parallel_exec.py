import multiprocessing

import copy
import numpy as np
import torch as to
import json
import os
import pathlib
import sys
import tqdm
import ray.tune as tune

import bayesopt
import config
import figures
import train
import util


ALL_ALGORITHMS = ['Random',
                  'Random_SteppedLR',
                  'Lorraine',
                  'Baydin',
                  'Ours_LR',
                  'Ours_LR_Momentum',
                  'Ours_HDLR_Momentum',
                  'DiffThroughOpt']


def generate_random_config_from_template(template):
    """Use the {hyperparameter: (min, max)} structure of `template` to produce
    an identically-structured nested dictionary with random values substituted.
    """
    config = {}
    for key, random_range in template.items():
        # Recurse into nested dictionaries
        if isinstance(random_range, dict):
            config[key] = generate_random_config_from_template(random_range)
        else:
            minimum, maximum = map(to.tensor, random_range)
            config[key] = to.rand(minimum.shape) * (maximum - minimum) + minimum
            if config[key].ndim == 0:
                config[key] = config[key].item()
            else:
                config[key] = config[key].tolist()
    return config


def natural_sgd_generator(scheduled_lr=True, ablation_weights=False):
    """Create a config override for a collection of randomised SGD
    hyperparameters.
    """
    # Generate uniform momentum and dampening in natural space, then transform
    random_ranges = {'config_dicts': {
        'network_weight': {
            'lr': (-6, -1),
            'weight_decay': (-7, -2),
            'momentum': (0, 1)}}}
    if scheduled_lr:
        random_ranges['config_dicts']['network_weight']['lr_multiplier'] = (0.95, 1.01)
    if ablation_weights:
        random_ranges['config_dicts']['model'] = {'initial_weights': ((-5, -5), (5, 5))}
    random_config = generate_random_config_from_template(random_ranges)

    # Transform momentum with an inverse sigmoid
    for key in ('momentum',):
        value = random_config['config_dicts']['network_weight'][key]
        random_config['config_dicts']['network_weight'][key] = (
            -np.log((1 / value) - 1)).item()

    return random_config


def sgd_from_directory(root_directory, scheduled_lr=False):
    """Exposes the unique SGD network_weight configs found in `root_directory`.
    """
    unique_configs = set()
    for run_directory in os.scandir(root_directory):
        if not run_directory.is_dir() or run_directory.name == 'cache':
            continue
        config_path = os.path.join(root_directory, run_directory.name, "config.json")
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)['config_dicts']['network_weight']
            if scheduled_lr:
                if 'lr_multiplier' not in config:
                    continue
            else:
                config.pop('lr_multiplier', None)
            config.pop('class', 'None')
            unique_configs.add(
                tuple(config.items()))

    full_configs = []
    for config in unique_configs:
        full_configs.append({'config_dicts': {
            'network_weight': dict(config)}})
    return iter(full_configs)


def run_parallel(num_workers,
                 num_repetitions,
                 override_generator=lambda: {},
                 algorithms=ALL_ALGORITHMS,
                 main_function=train.main):
    """Main execution function for managing parallel runs.
    """
    if algorithms:
        master_config_dicts = []
        for algorithm in algorithms:
            # Load one config for each algorithm
            sys.argv.extend(['-c', f'./configs/{algorithm}.yaml'])
            master_config_dicts.append(config.load_config())
            sys.argv = sys.argv[:-2]
    else:
        master_config_dicts = [None]
    if num_repetitions is None:
        total = None
    else:
        total = num_repetitions*len(master_config_dicts)
    with tqdm.tqdm(total=total,
                   position=0,
                   smoothing=0) as master_progress:
        master_progress.set_lock(multiprocessing.RLock())
        pool = multiprocessing.Pool(processes=num_workers,
                                    initializer=tqdm.tqdm.set_lock,
                                    initargs=(tqdm.tqdm.get_lock(),))
        if num_repetitions is None:
            iterator = override_generator()
        else:
            iterator = (override_generator() for _ in range(num_repetitions))
        for master_repetition_override in iterator:
            for algorithm_dict in master_config_dicts:
                repetition_override = copy.deepcopy(master_repetition_override)
                if (algorithm_dict is not None
                        and algorithm_dict['algorithm'] != 'Random_SteppedLR'):
                    # Remove lr_multiplier key if it exists
                    (repetition_override
                     .get('config_dicts', {})
                     .get('network_weight', {})
                     .pop('lr_multiplier', None))
                pool.apply_async(main_function,
                                 kwds={'config_dict': algorithm_dict,
                                       'config_override': repetition_override},
                                 callback=lambda *_: master_progress.update())
        pool.close()
        pool.join()


def ray_tune_run_pbt(num_workers,
                     num_repetitions,
                     name,
                     local_dir='/scratch/rmc78/ShortHorizonBias/runs'):
    """Main execution function for managing Population-Based Training runs
    using Ray Tune.
    """
    def clip_momentum(config):
        config['momentum'] = np.clip(config['momentum'], 0, 1-1e-6)
        return config

    scheduler = tune.schedulers.PopulationBasedTraining(
        time_attr="training_iteration",
        perturbation_interval=100,
        hyperparam_mutations={
            'lr': tune.uniform(-6, -1),
            'weight_decay': tune.uniform(-7, -2),
            'momentum': tune.uniform(0, 1)},
        custom_explore_fn=clip_momentum)

    master_config = config.load_config()

    tune.run(
        ray_tune_trainable,
        name=f"PBT {name}",
        scheduler=scheduler,
        metric='validation_loss',
        mode='min',
        stop={'training_iteration': 400*10},
        # stop=lambda trial_id, result: not np.isfinite(result['validation_loss']),
        checkpoint_score_attr='min-validation_loss',
        num_samples=num_repetitions,
        resources_per_trial={'cpu': 1, 'gpu': 1/num_workers},
        local_dir=local_dir,
        config={'master_config': master_config,
                'num_workers': num_workers},
        log_to_file=True,
    )


def ray_tune_run_asha(num_workers,
                      num_repetitions,
                      name,
                      local_dir='/scratch/rmc78/ShortHorizonBias/runs'):
    """Main execution function for managing Asynchronous HyberBand runs using
    Ray Tune.
    """

    scheduler = tune.schedulers.ASHAScheduler(
        time_attr='training_iteration',
        max_t=400*10,
    )

    master_config = config.load_config()

    tune.run(
        ray_tune_trainable,
        name=f"ASHA {name}",
        scheduler=scheduler,
        metric='validation_loss',
        mode='min',
        stop=lambda trial_id, result: not np.isfinite(result['validation_loss']),
        num_samples=num_repetitions,
        resources_per_trial={'cpu': 1, 'gpu': 1/num_workers},
        local_dir=local_dir,
        config={'master_config': master_config,
                'num_workers': num_workers,
                'lr': tune.uniform(-6, -1),
                'weight_decay': tune.uniform(-7, -2),
                'momentum': tune.uniform(0, 1-1e-6)},
        log_to_file=True,
    )


def ray_tune_trainable(config, checkpoint_dir=None):
    """Trainable function to start training one configuration under Ray Tune.
    """
    # Undo Ray's directory changing so our relative paths work
    os.chdir(
        pathlib.Path(__file__).parent.resolve())

    momentum = config['momentum']
    transformed_momentum = (-np.log((1 / momentum) - 1)).item()

    repetition_override = {
        '_ray_tune_config': True,
        'config_dicts': {
            'network_weight': {
                'lr': config['lr'],
                'weight_decay': config['weight_decay'],
                'momentum': transformed_momentum}}}
    if checkpoint_dir:
        repetition_override['load_state'] = os.path.join(checkpoint_dir,
                                                         'checkpoint.pt')

    master_config = copy.deepcopy(config['master_config'])
    if config['num_workers'] > 1:
        tune.utils.wait_for_gpu(target_util=1-(1/config.pop('num_workers')))
    train.main(config_dict=master_config,
               config_override=repetition_override)


def replicate_initialisations(root_directory):
    """Override generator which replicates the model initialisation of every
    run logged in `root_directory`.
    """
    run_directories = [entry.path for entry in os.scandir(root_directory)
                       if entry.is_dir()]
    directory_iter = iter(run_directories)
    return lambda: {'config_dicts':
               {'model':
                {'curvature_initialisation': next(directory_iter),
                 'initialise_weights_from_curvature_file': True}}}


def hyperparameter_ablation_initialisations(true_lookback=False,
                                            configs_per_setting=100):
    base_config = config.load_config()
    group_name = base_config.get('run_group_name', "")
    run_configs = [natural_sgd_generator()
                   for _ in range(configs_per_setting)]
    for update_interval in range(1, 10+1):
        for lookback_distance in range(1, 1+(update_interval if true_lookback else 10)):
            for run_config in run_configs:
                run_config = copy.deepcopy(run_config)
                run_config['config_dicts']['hyperparameter_wrapper'] = {
                    'hyperparameter_rollback_distance': lookback_distance}
                run_config['network_weight_steps'] = update_interval
                # Total step limit is governed by config, so no need to set
                # hyperparameter_steps
                run_config['run_group_name'] = group_name + f"/Update{update_interval}_Rollback{lookback_distance}"
                yield run_config


def configs_from_directory(root_directory,
                           modifier=lambda config: None,
                           condition=lambda config: True):
    """Read all the configs found in runs logged under `root_directory`."""
    for run_directory in os.scandir(root_directory):
        if not run_directory.is_dir() or run_directory.name == 'cache':
            continue
        config_path = os.path.join(root_directory, run_directory.name, "config.json")
        if not os.path.exists(config_path):
            continue
        with open(config_path, 'r') as config_file:
            run_config = json.load(config_file)
        modifier(run_config)
        if not condition(run_config):
            continue
        yield run_config


def diff_through_opt_configs_from_ablation_base(directory_prefix):
    """Read the original hyperparameter ablation study runs in
    `directory_prefix`*, and repeat equivalent runs using DiffThroughOpt.
    """
    base_config = config.load_config()
    group_name = base_config.get('run_group_name', "")
    for update_interval in range(1, 10+1):
        for lookback_distance in range(1, update_interval+1):
            run_configs = configs_from_directory(f"{directory_prefix}/Update{update_interval}_Rollback{lookback_distance}")
            for run_config in run_configs:
                run_config['algorithm'] = 'DiffThroughOpt'
                run_config['config_dicts']['hyperparameter_wrapper']['class'] = 'ExactHypergradientOptimiser'
                run_config['reset_loop_before_hyperparameter_step'] = run_config['config_dicts']['hyperparameter_wrapper'].pop('hyperparameter_rollback_distance')
                run_config['run_group_name'] = group_name + f"/Update{update_interval}_Rollback{lookback_distance}"
                yield run_config


def random_configs_from_ablation_base(directory_prefix):
    """Read the original hyperparameter ablation study runs in
    `directory_prefix`*, and repeat equivalent runs using Random.
    """
    # Use Update10 to give 4000 network weight updates, matching other
    # experiments. Rollback doesn't matter; we'll be ignoring it.
    run_configs = configs_from_directory(f"{directory_prefix}/Update10_Rollback5")
    for run_config in run_configs:
        run_config['algorithm'] = 'Random'
        run_config['validation_proportion'] = 0
        run_config['patch_optimiser'] = False
        run_config['config_dicts'].pop('hyperparameter')
        run_config['config_dicts'].pop('hyperparameter_wrapper')
        run_config.pop('optimised_hyperparameters')
        yield run_config


def iteration_id():
    """Generate a config dict for sequential identification only."""
    counter = 1
    master_config = config.load_config()
    logging_base = os.path.join(
        master_config.get('log_root', '.'),
        master_config.get('run_group_name', '.'))

    def configurator():
        nonlocal counter
        override = {'run_group_name': f"Repeat_{counter}",
                    'log_root': logging_base}
        log_folder = os.path.join(logging_base, override['run_group_name'])
        if not os.path.exists(log_folder):
            os.makedirs(log_folder)
        override['log_folder'] = log_folder
        counter += 1
        return override

    return configurator


def hypergradient_comparison_ablation_initialisations():
    """Return natural_sgd_generator() initialisations with additional settings
    configured for the production of comparable hypergradients.
    """
    for config_override in hyperparameter_ablation_initialisations(
            true_lookback=True, configs_per_setting=50):
        # Random seed from range of valid seeds
        (config_override
         .setdefault('config_dicts', {})
         .setdefault('model', {})
         ['random_seed']) = to.seed()
        # Need two hyperparameter steps so the first gets fully logged;
        # in order to properly log the initial state, we log _before_ each
        # hyperparameter update
        config_override['hyperparameter_steps'] = 2
        yield config_override


def fix_missing_hyperparameter_clipping(root_directory):
    """Repeat any runs in `root_directory` which were missing hyperparameter
    clipping.
    """
    no_lr_algorithms = ('Random', 'Lorraine', 'Ours_HDLR_Momentum')
    data = util.get_tags(root_directory)
    learning_rates = figures.parse_data_by_algorithm(data,
                                                     'Hyperparameter/Lr',
                                                     delete_nans=False,
                                                     add_random_batch=False,
                                                     exclude_algorithms=no_lr_algorithms)
    configs = figures.parse_data_by_algorithm(data,
                                              'config',
                                              delete_nans=False,
                                              add_random_batch=False)
    rerun_configs = []
    for algorithm, algorithm_data in learning_rates.items():
        for run_config, lr_history in zip(configs[algorithm], algorithm_data):
            if lr_history.max() > 1 or lr_history.min() < 1e-10:
                rerun_configs.append(run_config)
    rerun_configs.extend(configs['Ours_HDLR_Momentum'])

    print(f'Rerunning {len(rerun_configs)} configs.')
    for rerun_config in rerun_configs:
        rerun_config['hyperparameter_clipping'] = {'lr': [-10.0, 0.0]}
        yield rerun_config


if __name__ == '__main__':
    # Python <=3.8 uses a relative __file__; force it to be absolute
    __file__ = os.path.abspath(__file__)
    # run_parallel(num_workers=8,
    #              num_repetitions=200,
    #              override_generator=natural_sgd_generator,
    #              algorithms=ALL_ALGORITHMS + ['Random_Validation'],
    #              main_function=train.main)
    #
    # import signal
    # import pdb
    # signal.signal(signal.SIGUSR1, lambda *_: pdb.set_trace())
    ray_tune_run_pbt(num_workers=8,
                     num_repetitions=200)

"""Helper utilities and functions.
"""

import json
import glob
import os
import shutil
from pathlib import Path

import numpy as np
import torch as to
import tqdm
from tensorboard.backend.event_processing.event_accumulator import SCALARS
from tensorboard.backend.event_processing.event_multiplexer import EventMultiplexer


def load_tensorboard_data(directory, tags=None, ordered=False, penn_treebank=False):
    """Load the TensorBoard logs from `directory` containing `tags` into
    Python.
    """
    # Set SCALARS size_guidance to 0 to load every event
    accumulator = EventMultiplexer(size_guidance={SCALARS: 0})
    next_runs = {}
    non_first_runs = set()
    if ordered:
        for run_directory in sorted(os.scandir(directory),
                                    key=lambda x: x.name):
            if run_directory.name == 'cache' or not run_directory.is_dir():
                continue
            accumulator.AddRun(run_directory.path, name=run_directory.name)
    elif penn_treebank:
        # Special loading logic to merge split log files
        for algorithm_directory in os.scandir(directory):
            if not algorithm_directory.is_dir():
                continue
            if algorithm_directory.name == 'Ours_HDLR_Momentum':
                for run_directory in os.scandir(algorithm_directory):
                    if not run_directory.is_dir():
                        continue
                    run_files = sorted((log_file for log_file in os.scandir(run_directory)
                                        if log_file.name.startswith("events.out.tfevents")),
                                       key=lambda x: x.name)
                    # Fix root_name to be same style as others
                    root_name = os.path.join(algorithm_directory.name, run_directory.name)
                    accumulator.AddRun(run_files[0].path, name=root_name)
                    accumulator.AddRun(run_files[1].path)
                    next_runs[root_name] = run_files[1].path
                    non_first_runs.add(run_files[1].path)
            else:
                # Need to specify root name for paths to work properly
                accumulator.AddRunsFromDirectory(algorithm_directory.path,
                                                 name=algorithm_directory.name)
    else:
        accumulator.AddRunsFromDirectory(directory)
    accumulator.Reload()
    if tags is None:
        tags = [run_tags['scalars']
                for run_tags in accumulator.Runs().values()]

    for run_name, tags in zip(accumulator.Runs(), tags):
        if run_name in non_first_runs:
            continue
        for tag in tags:
            values = accumulator.Scalars(run_name, tag)
            if run_name in next_runs:
                next_values = accumulator.Scalars(next_runs[run_name], tag)
                # Adjust new timestamps to emulate last gap in previous log file
                time_adjustment = (next_values[0].wall_time - values[-1].wall_time
                                   - (values[-1].wall_time - values[-2].wall_time))
                for event_id, event in enumerate(next_values):
                    next_values[event_id] = event._replace(
                        wall_time=event.wall_time - time_adjustment)
                # The logging process which produced values will have an extra
                # call to train.Learner.log_now(end_of_pass=False), which is
                # duplicated by the first call in the process producing
                # next_values. So if this isn't an end-of-pass tag,
                # delete the last entry of values to remove the duplicate.
                if not tag.startswith('Last_'):
                    values = values[:-1]
                values.extend(next_values)
            yield run_name, tag, values


def decode_tensorboard_summary(directory):
    """Read the tensorboard summary logged at `file_path`, saving a
    JSON-decoded version to the same directory.
    """
    with open(os.path.join(directory,
                           'decoded_events.json'), 'w') as decoded_file:
        for run_name, tag, tag_data in load_tensorboard_data(directory):
            for event in tag_data:
                dict_data = {'wall_time': event.wall_time,
                             'step': event.step,
                             'value': event.value,
                             'run_name': run_name,
                             'key': tag}
                json.dump(dict_data,
                          decoded_file,
                          separators=(',', ':'))
                decoded_file.write('\n')


def get_tags(directory, *tags, rebuild_cache=False, **loading_kwargs):
    """Read the TensorBoard summary in `directory` and extract the trajectories
    of `tags`, stacking these into tensors. A cache of stacked tags is kept.
    """
    data = {'_name': [],
            'config': [],
            'first_timestamp': [],
            'last_timestamp': []}
    cache_prefix = os.path.join(directory, 'cache/')
    if rebuild_cache:
        reloaded_tags = None
    elif not tags and os.path.exists(cache_prefix):
        cached_files = glob.glob(
            os.path.join(cache_prefix, '**', '*.pt'),
            recursive=True)
        for cache_file in cached_files:
            cached_data = to.load(cache_file)
            # Strip prefix and '.pt'
            cached_key = cache_file[len(cache_prefix):-3]
            data[cached_key] = cached_data
        return data
    elif not tags:
        rebuild_cache = True
        reloaded_tags = None
    else:
        reloaded_tags = []
        for tag in tags:
            cache_path = os.path.join(directory, 'cache', tag + '.pt')
            if os.path.exists(cache_path):
                data[tag] = to.load(cache_path)
            else:
                reloaded_tags.append(tag)

    # Check for empty list so that None still gets interpreted as reading all
    # tags
    if reloaded_tags != []:
        for run_name, tag, tag_data in load_tensorboard_data(directory,
                                                             reloaded_tags,
                                                             **loading_kwargs):
            if tag not in data:
                data[tag] = []
                data[tag + '/wall_time'] = []
            data[tag].append(to.tensor([event.value for event in tag_data]))
            data[tag + '/wall_time'].append(
                to.tensor([event.wall_time for event in tag_data], dtype=to.float64))
            # Add reversed for efficiency
            if run_name not in reversed(data['_name']):
                data['_name'].append(run_name)
                data['first_timestamp'].append(float('inf'))
                data['last_timestamp'].append(0.0)
            data['first_timestamp'][-1] = min(data['first_timestamp'][-1],
                                              *(event.wall_time for event in tag_data))
            data['last_timestamp'][-1] = max(data['last_timestamp'][-1],
                                             *(event.wall_time for event in tag_data))
        for run_name in data['_name']:
            with open(os.path.join(directory, run_name, 'config.json'), 'r') as config_file:
                data['config'].append(json.load(config_file))

        if reloaded_tags is None:
            tag_iterator = data
        else:
            tag_iterator = reloaded_tags

        for tag in tag_iterator:
            cache_path = os.path.join(directory, 'cache', tag + '.pt')
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            to.save(data[tag], cache_path)

    return data


def nested_update(source_dict, update_dict):
    """Recursively update each level of `source_dict` with the contents of
    `update_dict`.
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in source_dict:
            nested_update(source_dict[key], value)
        else:
            source_dict[key] = value


def bootstrap_sample(data_length, num_datasets, num_samples=None):
    """Bootstrap sample, generating `num_datasets` sample sets of `num_samples`
    each, returning the indices of the sample.
    """
    if num_samples is None:
        num_samples = data_length
    return np.random.choice(data_length,
                            replace=True,
                            size=(num_datasets, num_samples))


def compute_full_dataset_loss(dataloader, model, loss_function):
    """Compute an average loss over all batches in `dataloader`, using the
    provided `model` and `loss_function`.
    """
    device = next(model.parameters()).device
    with to.no_grad():
        hidden_states = []
        losses = []
        for batch in dataloader:
            inputs, targets = [b.to(device) for b in batch]
            predictions = model(inputs, *hidden_states)
            if isinstance(predictions, tuple):
                predictions, hidden_states = predictions
            losses.append(
                loss_function(predictions, targets))
    return to.mean(
        to.stack(losses))


def get_unique_configs(root_directory):
    configs = set()
    for directory in os.scandir(root_directory):
        with open(os.path.join(directory.path, 'config.json'), 'r') as f:
            config = json.load(f)
            init = config['config_dicts']['network_weight']
            init.pop('lr_multiplier', None)
            configs.add(tuple(init.items()))
    return configs


def trim_configs(root_directory, target_num):
    """Randomly select `target_num` configurations in `root_directory` to
    retain, and separate the rest.
    """
    configs = get_unique_configs(root_directory)
    retained_config_indices = np.random.choice(len(configs),
                                               size=target_num,
                                               replace=False)
    all_configs = list(configs)
    retained_configs = set(all_configs[i]
                           for i in retained_config_indices)
    directories_to_move = []
    for directory in os.scandir(root_directory):
        with open(os.path.join(directory.path, 'config.json'), 'r') as f:
            config = json.load(f)
            init = config['config_dicts']['network_weight']
            init.pop('lr_multiplier', None)
            if tuple(init.items()) not in retained_configs:
                directories_to_move.append(directory)

    for directory in directories_to_move:
        shutil.move(directory.path,
                    os.path.join(root_directory, 'extra', directory.name))


def search_configs(root_directory, condition=lambda config: True):
    """Search all runs found in `root_directory`, returning the directories
    corresponding to runs where `condition` is met on their configs.
    """
    directories_found = []
    for directory in os.scandir(root_directory):
        if directory.name == 'cache' or not directory.is_dir():
            continue
        with open(os.path.join(directory, 'config.json'), 'r') as config_file:
            config = json.load(config_file)
            if condition(config):
                directories_found.append(directory)
    return directories_found


def interpolate_timestamps(data_values, data_times, num_timestamps):
    """Interpolate the `data_values` over `data_times` so that the data can all
    be represented with one array of `num_timestamps` timestamps.
    """
    # Keep second dimension when indexing
    data_elapsed_times = data_times - data_times[:, 0:1]
    interp_timestamps = np.linspace(data_elapsed_times.min(),
                                    data_elapsed_times.max(),
                                    num_timestamps)

    all_values = np.empty((len(data_values), num_timestamps))
    for run_id, (run_values, run_timestamps) in enumerate(zip(
            data_values, data_elapsed_times)):
        assert np.all(np.diff(run_timestamps) > 0)
        all_values[run_id] = np.interp(interp_timestamps,
                                       run_timestamps,
                                       run_values)

    return all_values, interp_timestamps


def flatten_config(config, prefix=''):
    """Transform the nested structure of `config` into a flat dictionary."""
    result = {}
    for key, value in config.items():
        if isinstance(value, dict):
            new_prefix = prefix + key + '.'
            result.update(
                flatten_config(value, prefix=new_prefix))
        else:
            result[prefix + key] = value
    return result


def common_configs(root_directory):
    """Combine all configs in `root_directory`, noting common values of the
    keys and those which vary."""
    configs = get_tags(root_directory)['config']
    common_keys = flatten_config(configs[0])
    optional_keys = {}
    different_keys = {}
    for key, value in common_keys.items():
        if isinstance(value, list):
            common_keys[key] = tuple(value)

    for config in configs:
        config = flatten_config(config)
        for key, value in config.items():
            if isinstance(value, list):
                value = tuple(value)

            if key in different_keys:
                different_keys[key].add(value)
                continue
            if key in optional_keys:
                if optional_keys[key] == value:
                    continue
                else:
                    different_keys[key] = set([optional_keys.pop(key),
                                               value])
                    continue
            if key in common_keys:
                if common_keys[key] == value:
                    continue
                else:
                    different_keys[key] = set([common_keys.pop(key),
                                               value])
            else:  # key not anywhere
                optional_keys[key] = value
        common_keys_to_remove = []
        for key in common_keys.keys():
            if key not in config:
                optional_keys[key] = common_keys[key]
                common_keys_to_remove.append(key)
        for key in common_keys_to_remove:
            del common_keys[key]
    return common_keys, optional_keys, different_keys

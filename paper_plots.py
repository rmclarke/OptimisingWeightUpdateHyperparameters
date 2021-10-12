"""Script generating all plots for paper."""
import os
import pickle
from contextlib import contextmanager
from itertools import chain
from functools import partial

import torch as to
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from cycler import cycler

import figures
import util


UCI_DATASETS = ['uci_boston',
                'uci_concrete',
                'uci_energy',
                'uci_kin8nm',
                'uci_naval',
                'uci_power',
                'uci_wine',
                'uci_yacht']
LARGE_SCALE_DATASETS = ['Fashion-MNIST',
                        'PennTreebank',
                        'CIFAR-10']
STANDALONE_LONGDIFFTHROUGHOPT_DATASETS = ['uci_energy_LongDiffThroughOpt_Standalone',
                                          'uci_kin8nm_LongDiffThroughOpt_Standalone',
                                          'uci_power_LongDiffThroughOpt_Standalone',
                                          'Fashion-MNIST_LongDiffThroughOpt_Standalone',]
ALGORITHM_LABELS = {
    'Random': "Random",
    'Random_SteppedLR': r"Random ($\times$ LR)",
    'Random_3Batched': "Random (3-batched)",
    'Lorraine': "Lorraine",
    'Baydin': "Baydin",
    'Ours_LR': r"Ours$^\text{WD+LR}$",
    'Ours_LR_Momentum': r"Ours$^\text{WD+LR+M}$",
    'Ours_HDLR_Momentum': r"Ours$^\text{WD+HDLR+M}$",
    'DiffThroughOpt': "Diff-through-Opt",
    'LongDiffThroughOpt': "Diff-through-Opt", # User must update label
    'BayesOpt': "Bayesian Optimisation"
}
NAN_THRESHOLDS = {
    'uci_boston': None,
    'uci_concrete': None,
    'uci_energy': None,
    'uci_kin8nm': None,
    'uci_naval': None,
    'uci_power': None,
    'uci_wine': None,
    'uci_yacht': None,
    'uci_energy_LongDiffThroughOpt_Standalone': None,
    'uci_energy_LongDiffThroughOpt_Full': None,
    'uci_kin8nm_LongDiffThroughOpt_Standalone': None,
    'uci_power_LongDiffThroughOpt_Standalone': None,
    'Fashion-MNIST': 1e3,
    'Fashion-MNIST_BatchNorm': 1e3,
    'Fashion-MNIST_LongDiffThroughOpt_Medium': 1e3,
    'Fashion-MNIST_LongDiffThroughOpt_Standalone': 1e3,
    'PennTreebank': 1e5,
    'CIFAR-10': None,
}


def metric(dataset):
    if dataset in (UCI_DATASETS
                   + STANDALONE_LONGDIFFTHROUGHOPT_DATASETS[:-1]
                   + ['uci_energy_LongDiffThroughOpt_Full']):
        return 'Unnormalised_Loss/Test'
    elif dataset == 'PennTreebank':
        return 'Perplexity/Test'
    elif dataset in (LARGE_SCALE_DATASETS + ['Fashion-MNIST_BatchNorm',
                                             'Fashion-MNIST_LongDiffThroughOpt_Medium',
                                             'Fashion-MNIST_LongDiffThroughOpt_Standalone',
                                             'Fashion-MNIST_Comparative_LongDiffThroughOpt']):
        return 'Loss/Test'
    else:
        raise ValueError(f"Don't know how to handle dataset {dataset}")


def xlabel(dataset, evolution):
    if dataset in (UCI_DATASETS
                   + STANDALONE_LONGDIFFTHROUGHOPT_DATASETS[:-1]
                   + ['uci_energy_LongDiffThroughOpt_Full']):
        label = 'Test MSE (Unnormalised)'
    elif dataset == 'PennTreebank':
        label = 'Test Perplexity'
    elif dataset in (LARGE_SCALE_DATASETS + ['Fashion-MNIST_BatchNorm',
                                             'Fashion-MNIST_LongDiffThroughOpt_Medium',
                                             'Fashion-MNIST_LongDiffThroughOpt_Standalone',
                                             'Fashion-MNIST_Comparative_LongDiffThroughOpt']):
        label = 'Test Cross-Entropy'
    else:
        raise ValueError(f"Don't know how to handle dataset {dataset}")

    if not evolution:
        label = 'Final ' + label
    return label


@contextmanager
def savefig(name):
    """Replace plt.show() with a custom function saving the figure to file."""
    with nofig():
        yield
        plt.savefig(f'./docs/ICLR2022/Figures/{name}.pdf')
        plt.tight_layout()


@contextmanager
def nofig():
    """Block all figure plotting."""
    original_show = plt.show
    plt.show = lambda: None
    yield
    plt.close()
    plt.show = original_show


@contextmanager
def paper_theme(exclude_algorithms=[],
                use_custom_cycler=True,
                use_bayesopt=False,
                use_long_diffthroughopt=False,
                long_diffthroughopt_standalone=False):
    """Configure matplotlib to plot in a standard theme."""
    base03 = '#002b36'
    base02 = '#073642'
    base01 = '#586e75'
    base00 = '#657b83'
    base0 = '#839496'
    base1 = '#93a1a1'
    base2 = '#eee8d5'
    base3 = '#fdf6e3'
    yellow = '#b58900'
    orange = '#cb4b16'
    red = '#dc322f'
    magenta = '#d33682'
    violet = '#6c71c4'
    blue = '#268bd2'
    cyan = '#2aa198'
    green = '#859900'

    design_spec = {
        'BayesOpt': {'color': 'k', 'linestyle': '-'},
        'LongDiffThroughOpt': {'color': 'k', 'linestyle': '--'},
        'Random': {'color': cyan, 'linestyle': '-'},
        'Random_SteppedLR': {'color': '#B2B2B2', 'linestyle': '-'},
        'Random_3Batched': {'color': '#B2B2B2', 'linestyle': '--'},
        'Lorraine': {'color': green, 'linestyle': '--'},
        'Baydin': {'color': yellow, 'linestyle': '-'},
        'Ours_LR': {'color': orange, 'linestyle': '--'},
        'Ours_LR_Momentum': {'color': red, 'linestyle': '-'},
        'Ours_HDLR_Momentum': {'color': magenta, 'linestyle': '--'},
        'DiffThroughOpt': {'color': violet, 'linestyle': '-'},
    }
    if not use_bayesopt:
        del design_spec['BayesOpt']
    if not (use_long_diffthroughopt or long_diffthroughopt_standalone):
        del design_spec['LongDiffThroughOpt']
    if long_diffthroughopt_standalone:
        del design_spec['Random_SteppedLR']
        del design_spec['Random_3Batched']
        del design_spec['Lorraine']
        del design_spec['Baydin']
        del design_spec['Ours_LR']
        del design_spec['Ours_HDLR_Momentum']
    for algorithm in exclude_algorithms:
        design_spec.pop(algorithm, None)

    cycles = {}
    for algorithm, spec in design_spec.items():
        for key, value in spec.items():
            cycles.setdefault(key, []).append(value)
    custom_cycler = cycler(**cycles)

    with plt.style.context('Solarize_Light2'):
        if use_custom_cycler:
            plt.rc('axes', prop_cycle=custom_cycler)
        plt.gcf().set_facecolor('white')
        yield


def sensitivity_study():
    """Construct heatmap plots showing the sensitivity of final performance to
    the choice of update interval and lookback distance.
    """
    value_scale = 1000
    data_metric = metric('uci_energy')
    data_random = util.get_tags(
        './runs/ICLR Sensitivity UCI_Energy Random', data_metric)
    random_value = to.stack(data_random[data_metric])[:, -1].nanmedian().max() * value_scale
    global_max = -float('inf')
    for root_dir in ('./runs/ICLR Sensitivity UCI_Energy Ours_LR_Momentum',
                     './runs/ICLR Sensitivity UCI_Energy DiffThroughOpt'):
        for sub_dir in os.scandir(root_dir):
            data = util.get_tags(sub_dir.path, data_metric)
            global_max = max(global_max,
                             to.stack(data[data_metric])[:, -1].nanmedian().max() * value_scale)
    normaliser = Normalize(vmin=0, vmax=global_max)

    with savefig('Sensitivity_OursLRMomentum'):
        figures.plot_toy_ablation_heatmap(
            './runs/ICLR Sensitivity UCI_Energy Ours_LR_Momentum',
            normaliser=normaliser,
            num_format='{:.0f}',
            value_scale=value_scale)
    with savefig('Sensitivity_DiffThroughOpt'):
        figures.plot_toy_ablation_heatmap(
            './runs/ICLR Sensitivity UCI_Energy DiffThroughOpt',
            normaliser=normaliser,
            num_format='{:.0f}',
            value_scale=value_scale)

    plt.imshow([[random_value]], norm=normaliser)
    text = plt.text(0, 0, f"{random_value:.0f}", ha='center', va='center')
    if normaliser(random_value) < normaliser(global_max) / 2:
        text.set_color('w')
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.gcf().set_size_inches(1, 1)
    plt.tight_layout()
    with savefig('Sensitivity_Random'):
        plt.show()


def hypergradient_comparison():
    """Construct heatmap plots showing the relative error in hypergradients
    between ours and exact methods.
    """
    def data_extractor(data_dict, key):
        config_data = figures.parse_data_by_algorithm(
            data_dict, 'config', delete_nans=False, add_random_batch=False)
        hyperparameter_data = figures.parse_data_by_algorithm(
            data_dict, key, delete_nans=False, add_random_batch=False)

        true_hyperparameters = hyperparameter_data['DiffThroughOpt']
        config_map = []
        for true_config in config_data['DiffThroughOpt']:
            for index, our_config in enumerate(config_data['Ours_LR_Momentum']):
                if (our_config['config_dicts']['model']['random_seed']
                          == true_config['config_dicts']['model']['random_seed']):
                    config_map.append(index)
        our_hyperparameters = hyperparameter_data['Ours_LR_Momentum'][config_map]

        true_updates = true_hyperparameters[:, 1] - true_hyperparameters[:, 0]
        our_updates = our_hyperparameters[:, 1] - our_hyperparameters[:, 0]
        hypergradient_errors = to.abs((true_updates - our_updates) / true_updates)
        return hypergradient_errors.numpy()

    with savefig('Hypergradients_Lr'):
        figures.plot_toy_ablation_heatmap(
            './runs/ICLR HypergradientComparison',
            data_extractor=partial(data_extractor, key='Last_Hyperparameter/Lr'),
            aggregator=np.nanmean,
            num_format='{:.1%}')
    with savefig('Hypergradients_Weight_Decay'):
        figures.plot_toy_ablation_heatmap(
            './runs/ICLR HypergradientComparison',
            data_extractor=partial(data_extractor, key='Last_Hyperparameter/Weight_Decay'),
            aggregator=np.nanmean,
            num_format='{:.0%}')


def loss_figures():
    """Construct plots of the evolving losses on all datasets (with error
    envelopes).
    """
    table_datasets = set(('uci_energy', 'uci_kin8nm', 'uci_power',
                          'Fashion-MNIST', 'PennTreebank', 'CIFAR-10',
                          'Fashion-MNIST_BatchNorm', 'Fashion-MNIST_LongDiffThroughOpt_Medium',
                          *STANDALONE_LONGDIFFTHROUGHOPT_DATASETS,
                          'uci_energy_LongDiffThroughOpt_Full'))
    for dataset in (UCI_DATASETS
                    + LARGE_SCALE_DATASETS
                    + STANDALONE_LONGDIFFTHROUGHOPT_DATASETS
                    + ['uci_energy_LongDiffThroughOpt_Full']
                    + ['Fashion-MNIST_BatchNorm', 'Fashion-MNIST_LongDiffThroughOpt_Medium']):
        print(f'    {dataset}...')
        print('        Collecting data...')
        data = util.get_tags(f'./runs/ICLR {dataset}')

        print('        Loss envelope...')
        with savefig(f'Envelope_{dataset}'), paper_theme(
                long_diffthroughopt_standalone=(
                    dataset in STANDALONE_LONGDIFFTHROUGHOPT_DATASETS)):
            bootstrapped_data = figures.plot_evolution_envelope(
                data,
                metric(dataset),
                legend=False,
                nearly_nan_threshold=NAN_THRESHOLDS[dataset])
            plt.xlabel("Network Weight Update Step")
            plt.ylabel(xlabel(dataset, evolution=True))
            plt.xlim(0, tuple(bootstrapped_data.values())[0].shape[-1])
            plt.gcf().set_size_inches(3, 3)
            plt.tight_layout()

        if dataset in table_datasets:
            source_data = figures.parse_data_by_algorithm(
                data,
                metric(dataset),
                add_random_batch=('Random_3Batched' in bootstrapped_data))
            write_statistics(bootstrapped_data,
                             source_data,
                             dataset,
                             'AverageResults')


def error_figures():
    """Prepare data for a table of errors on classification datasets."""
    datasets = ('Fashion-MNIST',
                'CIFAR-10',
                'Fashion-MNIST_LongDiffThroughOpt_Medium',
                'Fashion-MNIST_LongDiffThroughOpt_Standalone')
    for dataset in datasets:
        print(f'    {dataset}...')
        print('        Collecting data...')
        data = util.get_tags(f'./runs/ICLR {dataset}')
        data['Error/Test'] = [1 - accuracy
                              for accuracy in data['Accuracy/Test']]
        with nofig():
            print('        Bootstrap sampling data...')
            bootstrapped_data = figures.plot_evolution_envelope(
                data, 'Error/Test')
        source_data = figures.parse_data_by_algorithm(
            data,
            'Error/Test',
            add_random_batch=('Random_3Batched' in bootstrapped_data))
        write_statistics(bootstrapped_data,
                         source_data,
                         dataset,
                         'AverageErrors')


def write_statistics(bootstrapped_data, original_data, dataset, statistic_name):
    """Compute statistics from `bootstrapped_data` for `dataset` and save to
    file for later plotting.
    """
    print('        Writing statistics...')
    statistics = {}
    # Construct table in transpose first
    for algorithm, algorithm_data in bootstrapped_data.items():
        mean_set = np.array([np.nanmean(sample_set[:, -1])
                             for sample_set in algorithm_data])
        median_set = np.array([np.nanmedian(sample_set[:, -1])
                               for sample_set in algorithm_data])
        best_value = np.nanmin(original_data[algorithm])
        if dataset.startswith('uci_kin8nm'):
            mean_set *= 1000
            median_set *= 1000
            best_value *= 1000

        statistics[algorithm] = {}
        statistics[algorithm]['mean_value'] = np.nanmean(mean_set)
        statistics[algorithm]['mean_error'] = np.nanstd(mean_set)
        statistics[algorithm]['median_value'] = np.nanmean(median_set)
        statistics[algorithm]['median_error'] = np.nanstd(median_set)
        statistics[algorithm]['best_value'] = best_value

    statistic_path = f'./docs/ICLR2022/Figures/{statistic_name}_{dataset}.pkl'
    with open(statistic_path, 'wb') as statistic_file:
        pickle.dump(statistics, statistic_file)


def write_all_tables():
    """Pass each table specification into write_table() to construct all
    tables.
    """
    table_specs = {
        'AverageResults_UCI': {
            'datasets': ('uci_energy', 'uci_kin8nm', 'uci_power'),
            'statistic': 'AverageResults'},
        'AverageResults_LargeScale': {
            'datasets': ('Fashion-MNIST', 'PennTreebank', 'CIFAR-10'),
            'statistic': 'AverageResults'},
        'AverageResults_Fashion-MNIST': {
            'datasets': ('Fashion-MNIST', 'Fashion-MNIST_BatchNorm'),
            'statistic': 'AverageResults'},
        'AverageErrors_LargeScale': {
            'datasets': ('Fashion-MNIST', 'CIFAR-10'),
            'statistic': 'AverageErrors'},

        'AverageMixed_Fashion-MNIST_Solo': {
            'datasets': ('Fashion-MNIST', 'Fashion-MNIST'),
            'statistic': ('AverageResults', 'AverageErrors')},
        'AverageMixed_Fashion-MNIST_LongDiffThroughOpt_Medium': {
            'datasets': (('Fashion-MNIST', 'Fashion-MNIST_LongDiffThroughOpt_Medium'),
                         ('Fashion-MNIST', 'Fashion-MNIST_LongDiffThroughOpt_Medium')),
            'statistic': (('AverageResults', 'AverageResults'),
                          ('AverageErrors', 'AverageErrors'))},

        'AverageResults_UCI_LongDiffThroughOpt_Standalone': {
            'datasets': ('uci_energy_LongDiffThroughOpt_Standalone',
                         'uci_kin8nm_LongDiffThroughOpt_Standalone',
                         'uci_power_LongDiffThroughOpt_Standalone'),
            'statistic': 'AverageResults'},
        'AverageResults_UCI_Energy_LongDiffThroughOpt_Full': {
            'datasets': (('uci_energy', 'uci_energy_LongDiffThroughOpt_Full'),),
            'statistic': (('AverageResults', 'AverageResults'),)},
        'AverageMixed_Fashion-MNIST_LongDiffThroughOpt_Standalone': {
            'datasets': ('Fashion-MNIST_LongDiffThroughOpt_Standalone', 'Fashion-MNIST_LongDiffThroughOpt_Standalone'),
            'statistic': ('AverageResults', 'AverageErrors')},

        'AverageResults_UCI_BayesOpt': {
            'datasets': ('uci_energy_BayesOpt', 'uci_kin8nm_BayesOpt', 'uci_power_BayesOpt'),
            'statistic': 'AverageResults',
            'bold_best': False},
        'AverageResults_Fashion-MNIST_BayesOpt': {
            'datasets': ('Fashion-MNIST_BayesOpt',),
            'statistic': 'AverageResults',
            'bold_best': False},
        'AverageResults_Fashion-MNIST_NonBayesOpt': {
            'datasets': ('Fashion-MNIST',),
            'statistic': 'AverageResults'},
    }
    for table_name, table_spec in table_specs.items():
        print(f'    {table_name}...')
        write_table(table_spec['datasets'],
                    table_name,
                    table_spec['statistic'],
                    table_spec.get('bold_best', True))


def write_table(table_datasets, table_name, statistic_names, bold_best=True):
    """Use the `bootstrapped_data` statistics to produce a results table over
    `table_datasets`.
    """
    if not isinstance(statistic_names, (tuple, list)):
        statistic_names = [statistic_names for _ in table_datasets]
    value_blocks = []
    error_blocks = []
    best_blocks = []
    for block_datasets, block_statistic_names in zip(table_datasets, statistic_names):
        if not isinstance(block_datasets, (tuple, list)):
            block_datasets = [block_datasets]
            block_statistic_names = [block_statistic_names]
        statistics = {}
        for dataset, statistic_name in zip(block_datasets, block_statistic_names):
            statistic_path = f'./docs/ICLR2022/Figures/{statistic_name}_{dataset}.pkl'
            with open(statistic_path, 'rb') as statistic_file:
                statistics.update(pickle.load(statistic_file))
        # Move LongDiffThroughOpt to the end if it's here
        if 'LongDiffThroughOpt' in statistics:
            statistics['LongDiffThroughOpt'] = statistics.pop('LongDiffThroughOpt')

        value_block = np.array([[algorithm_data['mean_value'],
                                 algorithm_data['median_value']]
                                for algorithm_data in statistics.values()])
        error_block = np.array([[algorithm_data['mean_error'],
                                 algorithm_data['median_error']]
                                for algorithm_data in statistics.values()])
        best_block = np.array([[algorithm_data['best_value']]
                               for algorithm_data in statistics.values()])
        value_blocks.append(value_block)
        error_blocks.append(error_block)
        best_blocks.append(best_block)
    table_values = np.concatenate(value_blocks, axis=1)
    table_errors = np.concatenate(error_blocks, axis=1)
    table_bests = np.concatenate(best_blocks, axis=1)

    best_values = np.nanargmin(table_values, axis=0)
    best_bests = np.nanmin(table_bests, axis=0)
    best_thresholds = np.take_along_axis(table_values + table_errors,
                                         best_values[None, :],
                                         axis=0).squeeze()

    with open(f'./docs/ICLR2022/Figures/{table_name}.tex', 'w') as table:
        for algorithm, algorithm_values, algorithm_errors, algorithm_bests in zip(
                statistics.keys(), table_values, table_errors, table_bests):
            algorithm_name = ALGORITHM_LABELS[algorithm]
            if algorithm == 'LongDiffThroughOpt':
                if 'LongDiffThroughOpt_Standalone' in table_name:
                    algorithm_name = 'Long ' + algorithm_name
                elif 'LongDiffThroughOpt_Medium' in table_name:
                    algorithm_name = 'Medium ' + algorithm_name
                elif 'LongDiffThroughOpt_Full' in table_name:
                    algorithm_name = 'Full ' + algorithm_name
                else:
                    raise ValueError(f"Can't handle table name {table_name}")
            table.write(algorithm_name)
            # Only write best once both mean and median have been written
            ready_for_best = False
            best_iter = iter(algorithm_bests)
            best_best_iter = iter(best_bests)
            for value, error, best_threshold in zip(
                    algorithm_values, algorithm_errors, best_thresholds):
                # Robustly round error to 1 significant figure
                error_str = np.format_float_positional(
                    error, precision=1, fractional=False, trim='-')
                # Ints can't be NaN, so must check this before conversion
                useful_precision = -np.floor(np.log10(
                    float('{:.1g}'.format(error))))
                if np.isnan(useful_precision):
                    table.write(r' & \multicolumn{2}{c}{---}')
                else:
                    useful_precision = useful_precision.astype(int)
                    rounded_value = round(value, useful_precision)
                    # Skip float formatting if we actually now have a rounded
                    # integer, so don't need to worry about precisions
                    if error < 1:
                        value_str = ('{{:.{}f}}'
                                     .format(useful_precision)
                                     .format(rounded_value))
                    else:
                        value_str = str(int(rounded_value))
                    if rounded_value <= best_threshold and bold_best:
                        value_str = r'\bfseries ' + value_str
                    table.write(f' & {value_str} & $\\pm\\,${error_str}')
                if ready_for_best:
                    best_value = next(best_iter)
                    best_best = next(best_best_iter)
                    best_str = '{:#.3g}'.format(best_value)
                    # Remove any unneeded decimal points
                    best_str = best_str.rstrip('.')
                    if bold_best and best_value <= best_best:
                        best_str = r'\bfseries ' + best_str
                    table.write(f' & {best_str}')
                ready_for_best = not ready_for_best
            table.write(r'\\' + '\n')


def cdfs(force_fashion_mnist_data=None):
    """Plot empirical CDFs of the final test losses of all datasets."""
    for dataset in (UCI_DATASETS
                    + LARGE_SCALE_DATASETS
                    + STANDALONE_LONGDIFFTHROUGHOPT_DATASETS
                    + ['uci_energy_LongDiffThroughOpt_Full']
                    + ['Fashion-MNIST_BatchNorm']):
        if force_fashion_mnist_data:
            dataset = 'Fashion-MNIST_LongDiffThroughOpt_Medium'
            print(f'    {dataset} loss...')
            data = force_fashion_mnist_data
        else:
            print(f'    {dataset} loss...')
            data = util.get_tags(f'./runs/ICLR {dataset}')
        if dataset == 'uci_energy_LongDiffThroughOpt_Full':
            base_data = util.get_tags('./runs/ICLR uci_energy')
            for key, value in data.items():
                value.extend(base_data[key])
        with savefig(f'CDF_{dataset}'), paper_theme(
                use_long_diffthroughopt=(
                    bool(force_fashion_mnist_data)
                    or dataset == 'uci_energy_LongDiffThroughOpt_Full'),
                long_diffthroughopt_standalone=(
                    dataset in STANDALONE_LONGDIFFTHROUGHOPT_DATASETS)):
            figures.plot_final_cdfs(
                data,
                metric(dataset),
                nearly_nan_threshold=NAN_THRESHOLDS[dataset],
                add_random_batch=(
                    dataset not in (STANDALONE_LONGDIFFTHROUGHOPT_DATASETS
                                    + ['uci_energy_LongDiffThroughOpt_Full'])))
            plt.xlabel(xlabel(dataset, evolution=False))
            plt.ylabel('Empirical CDF')
            plt.xscale('log')
            plt.ylim(0, 1)
            plt.gcf().set_size_inches(3, 3)
            plt.tight_layout()
        if force_fashion_mnist_data:
            break

    for dataset in ('Fashion-MNIST',
                    'CIFAR-10',
                    'Fashion-MNIST_BatchNorm',
                    'Fashion-MNIST_LongDiffThroughOpt_Standalone'):
        if force_fashion_mnist_data:
            dataset = 'Fashion-MNIST_LongDiffThroughOpt_Medium'
            print(f'    {dataset} error...')
            data = force_fashion_mnist_data
        else:
            print(f'    {dataset} error...')
            data = util.get_tags(f'./runs/ICLR {dataset}')
        data['Error/Test'] = [1 - accuracy for accuracy in data['Accuracy/Test']]
        with savefig(f'CDF_Error_{dataset}'), paper_theme(
                use_long_diffthroughopt=bool(force_fashion_mnist_data),
                long_diffthroughopt_standalone=(
                    dataset in STANDALONE_LONGDIFFTHROUGHOPT_DATASETS)):
            figures.plot_final_cdfs(data,
                                    'Error/Test',
                                    add_random_batch=(dataset not in STANDALONE_LONGDIFFTHROUGHOPT_DATASETS))
            plt.xlabel("Test Error")
            plt.ylabel('Empirical CDF')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.gcf().set_size_inches(3, 3)
            plt.tight_layout()
        if force_fashion_mnist_data:
            break


def uci_sample_learning_rates():
    """Plot sample learning rate evolutions for UCI datasets for a (fixed)
    random run index.
    """
    no_lr_algorithms = ('Random',
                        'Random_Validation',
                        'Lorraine',
                        'Ours_HDLR_Momentum',
                        'BayesOpt')
    for dataset in UCI_DATASETS:
        print(f'    {dataset}...')
        run_ids = {algorithm: float('nan') for algorithm in ALGORITHM_LABELS.keys()
                   if algorithm not in ('Random_3Batched', 'LongDiffThroughOpt')}
        data = util.get_tags(f'./runs/ICLR {dataset}')
        ids_without_lr_log = np.array([config['algorithm'] in no_lr_algorithms
                                       for config in data['config']])
        
        # Magic number for determinism
        target_init = data['config'][167]['config_dicts']['network_weight'].copy()
        target_init.pop('lr_multiplier', None)

        for config_id, config in enumerate(data['config']):
            trial_init = config['config_dicts']['network_weight'].copy()
            trial_init.pop('lr_multiplier', None)
            if trial_init == target_init:
                if config['algorithm'] not in no_lr_algorithms:
                    # Convert config index to Hyperparameter/Lr index
                    config_id -= ids_without_lr_log[:config_id].sum()
                run_ids[config['algorithm']] = config_id

        with savefig(f'LR_{dataset}'), paper_theme(
                exclude_algorithms=['Random_3Batched', 'Ours_HDLR_Momentum']):
            for algorithm, run_id in run_ids.items():
                if algorithm in no_lr_algorithms:
                    if algorithm in ('Ours_HDLR_Momentum',
                                     'BayesOpt',
                                     'Random_Validation'):
                        continue
                    else:
                        run_config = data['config'][run_id]
                        total_steps = (run_config['network_weight_steps']
                                       * run_config['hyperparameter_steps'])
                        lr_value = 10**run_config['config_dicts']['network_weight']['lr']
                        plt.plot([0, total_steps],
                                 [lr_value, lr_value])
                else:
                    plt.plot(data['Hyperparameter/Lr'][run_id])

            plt.xlabel('Network Weight Update Step')
            plt.ylabel('Learning Rate')
            plt.xlim(0, total_steps)
            plt.yscale('log')
            plt.ylim(1e-6, None)
            plt.gcf().set_size_inches(3, 3)
            plt.tight_layout()


def large_scale_runtimes():
    """Construct stacked violin plots of experimental runtimes for large-scale
    datasets.
    """
    for dataset in ("Fashion-MNIST", "CIFAR-10", "PennTreebank"):
        data = util.get_tags(f'./runs/ICLR {dataset}')
        with savefig(f'Runtime_{dataset}'), paper_theme(exclude_algorithms=['Random_3Batched']):
            figures.plot_runtime_violins(
                data, exclude_algorithms=['Random_Validation'])
            plt.xticks([], [])
            plt.ylabel('Experiment Duration (s)')
            plt.gcf().set_size_inches(3, 3)
            plt.tight_layout()


def proof_of_concept():
    """Construct a proof of concept plot showing hyperparameter evolutions"""
    data = util.get_tags('./runs/ICLR ProofOfConcept_Revised')
    with savefig('ProofOfConcept_uci_energy'), paper_theme(use_custom_cycler=False):
        plt.xlabel("Learning Rate")
        plt.ylabel("Weight Decay")
        plt.xscale('log')
        plt.yscale('log')
        figures.plot_hyperparameter_evolution(
            data,
            heatmap_algorithm='Random',
            trajectory_algorithm='Ours_LR',
            hyperparameter_x='lr',
            hyperparameter_y='weight_decay',
            metric=metric('uci_energy'),
            num_trajectories=20,
            hyperparameter_x_exclude_algorithms=('Random', 'Lorraine', 'Ours_HDLR_Momentum'),
            hyperparameter_y_exclude_algorithms=('Random', 'Random_SteppedLR', 'Baydin'),
            hyperparameter_x_init_transform=lambda x: 10**x,
            hyperparameter_y_init_transform=lambda y: 10**y,
            metric_transform=lambda z: np.log10(z))
        plt.gca().collections[0].colorbar.set_label("Final Test MSE (No HPO)")
        for line in plt.gca().get_lines():
            line.set_linewidth(0.5)
        plt.grid(False)
        plt.gcf().set_size_inches(4.5, 4.5)
        plt.tight_layout()

        # NaN hatching; must be last to avoid auto-scaling
        # Solarized Base1 edgecolor
        plt.autoscale(False)
        plt.fill_between([1e-7, 1e1], 1e-8, 1e1, hatch='xx', edgecolor='#93a1a1',
                         facecolor="none", zorder=-1000)


def best_performance_over_time():
    """Plot the evolution of best performance yet found with runtime."""
    bayesopt_datasets = ('uci_energy', 'uci_kin8nm', 'uci_power', 'Fashion-MNIST')
    datasets = (*bayesopt_datasets,
                *STANDALONE_LONGDIFFTHROUGHOPT_DATASETS,
                'uci_energy_LongDiffThroughOpt_Full',
                'Fashion-MNIST_Comparative_LongDiffThroughOpt')
    for dataset in datasets:
        print(f'    {dataset}...')
        print('        Collecting data...')
        if dataset == 'Fashion-MNIST_Comparative_LongDiffThroughOpt':
            data = util.get_tags('./runs/ICLR Fashion-MNIST')
            long_data = util.get_tags('./runs/ICLR Fashion-MNIST_LongDiffThroughOpt_Medium')
            for key, value in data.items():
                value.extend(long_data[key])
        elif dataset == 'uci_energy_LongDiffThroughOpt_Full':
            data = util.get_tags('./runs/ICLR uci_energy_LongDiffThroughOpt_Full')
            long_data = util.get_tags('./runs/ICLR uci_energy')
            for key, value in data.items():
                value.extend(long_data[key])
        else:
            data = util.get_tags(f'./runs/ICLR {dataset}')
        dataset_metric = metric(dataset)
        bayesopt = (dataset in bayesopt_datasets)

        if bayesopt:
            bayesopt_data = []
            for subdir in os.scandir(f'./runs/ICLR BayesOpt_{dataset}'):
                bo_repetition_data = util.get_tags(subdir.path, ordered=True)
                bo_repetition_data[dataset_metric] = to.cat(bo_repetition_data[dataset_metric])
                bo_repetition_data[dataset_metric + '/wall_time'] = to.cat(bo_repetition_data[dataset_metric + '/wall_time'])
                bayesopt_data.append(bo_repetition_data)
            data[dataset_metric].extend(
                [repetition[dataset_metric] for repetition in bayesopt_data])
            data[dataset_metric + '/wall_time'].extend(
                [repetition[dataset_metric + '/wall_time'] for repetition in bayesopt_data])
            # Ensure figures.parse_data_by_algorithm() works
            data['config'].extend([{'algorithm': 'BayesOpt'}
                                   for _ in bayesopt_data])

        for run_id, run_metric in enumerate(data[dataset_metric]):
            # Allow NaNs to be overwritten
            run_metric[to.isnan(run_metric)] = float('inf')
            data[dataset_metric][run_id] = np.minimum.accumulate(run_metric)

        print('        HPO curve...')
        with savefig(f'HPOCurves_{dataset}'), paper_theme(
                use_bayesopt=bayesopt, exclude_algorithms=['Random_3Batched'],
                use_long_diffthroughopt=(
                    dataset in ('Fashion-MNIST_Comparative_LongDiffThroughOpt',
                                'uci_energy_LongDiffThroughOpt_Full')),
                long_diffthroughopt_standalone=(
                    dataset in STANDALONE_LONGDIFFTHROUGHOPT_DATASETS)):
            bootstrapped_data = figures.plot_evolution_envelope(
                data,
                dataset_metric,
                legend=False,
                wall_time=True,
                ignore_algorithms=['Random_Validation'])
            plt.xscale('log')
            plt.ylabel(xlabel(dataset, evolution=True))
            plt.gcf().set_size_inches(3, 3)
            plt.tight_layout()

        if bayesopt:
            source_data = figures.parse_data_by_algorithm(
                data, metric(dataset), add_random_batch=False)
            write_statistics({'BayesOpt': bootstrapped_data['BayesOpt']},
                             {'BayesOpt': source_data['BayesOpt']},
                             dataset + '_BayesOpt',
                             'AverageResults')


def long_horizon_fashion_mnist_cdfs():
    """Construct Fashion-MNIST CDFs including the long-horizon DiffThroughOpt
    results.
    """
    print('    Collecting data...')
    data = util.get_tags('./runs/ICLR Fashion-MNIST')
    long_data = util.get_tags('./runs/ICLR Fashion-MNIST_LongDiffThroughOpt_Medium')
    dataset_metric = metric('Fashion-MNIST')
    data[dataset_metric].extend(long_data[dataset_metric])
    data[dataset_metric + '/wall_time'].extend(long_data[dataset_metric + '/wall_time'])
    data['Accuracy/Test'].extend(long_data['Accuracy/Test'])
    data['config'].extend(long_data['config'])

    print('    > cdf...')
    cdfs(force_fashion_mnist_data=data)


if __name__ == '__main__':
    functions = (
        sensitivity_study,
        hypergradient_comparison,
        cdfs,
        long_horizon_fashion_mnist_cdfs,
        loss_figures,
        error_figures,
        best_performance_over_time,
        write_all_tables,
        uci_sample_learning_rates,
        large_scale_runtimes,
        proof_of_concept,
    )
    for function in functions:
        print(f"Running {function.__name__}...")
        function()

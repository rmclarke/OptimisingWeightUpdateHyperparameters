"""Reference script for running all experiments."""

import logging
import multiprocessing
import os
import subprocess
import textwrap
from functools import partial

# import parallel_exec as pe
# import train
import util


COMMANDS = {}
RUNTIME_SENSITIVE_COMMANDS = (
    'ICLR uci_energy',
    'ICLR uci_kin8nm',
    'ICLR uci_power',
    'ICLR Fashion-MNIST',
    # 'ICLR CIFAR-10',
    # 'ICLR PennTreebank',
    'ICLR BayesOpt_fashion_mnist',
    'ICLR BayesOpt_uci_energy',
    'ICLR BayesOpt_uci_kin8nm',
    'ICLR BayesOpt_uci_power',
    'ICLR uci_energy_LongDiffThroughOpt_Standalone',
    'ICLR uci_kin8nm_LongDiffThroughOpt_Standalone',
    'ICLR uci_power_LongDiffThroughOpt_Standalone',
    'ICLR Fashion-MNIST_LongDiffThroughOpt_Standalone',
    'ICLR Fashion-MNIST_LongDiffThroughOpt_Medium',
    'ICLR uci_energy_LongDiffThroughOpt_Full',
    'ASHA Random_TrainingSetOnly',
    'ASHA Ours_LR_Momentum',
    'PBT Random_TrainingSetOnly',
    'PBT Ours_LR_Momentum',
)


# NOTE: The following configurations are run on a separate SLURM cluster, so
# are not run from this this file. We include commented-out exemplar commands
# which should be equivalent to the experiments we actually ran:
# 'ICLR CIFAR-10'
# 'ICLR PennTreebank'


# Use an absolute path for ASHA/PBT
log_root = '/scratch/rmc78/ShortHorizonBias/runs_repeat/'
cuda_device = 0


# 'ICLR uci_boston'
# 'ICLR uci_concrete'
# 'ICLR uci_energy'
# 'ICLR uci_kin8nm'
# 'ICLR uci_naval'
# 'ICLR uci_power'
# 'ICLR uci_wine'
# 'ICLR uci_yacht'
for dataset in ('uci_boston',
                'uci_concrete',
                'uci_energy',
                'uci_kin8nm',
                'uci_naval',
                'uci_power',
                'uci_wine',
                'uci_yacht'):
    python_cmd = """\
    import parallel_exec as pe
    import train
    pe.run_parallel(num_workers=8,
                    num_repetitions=200,
                    # lr_multiplier is stripped for all algorithms other than
                    # Random_SteppedLR
                    override_generator=pe.natural_sgd_generator,
                    algorithms=["Random",
                                "Random_SteppedLR",
                                "Lorraine",
                                "Baydin",
                                "Ours_LR",
                                "Ours_LR_Momentum",
                                "Ours_HDLR_Momentum",
                                "DiffThroughOpt",
                                "Random_Validation"],
                    main_function=train.main)"""
    python_cmd = textwrap.dedent(python_cmd)
    shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/{dataset}.yaml -g 'ICLR {dataset}' -l {log_root}"
    COMMANDS[f'ICLR {dataset}'] = shell_cmd
    # subprocess.run(shell_cmd, shell=True)


# 'ICLR Fashion-MNIST'
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=8,
                num_repetitions=100,
                # lr_multiplier is stripped for all algorithms other than
                # Random_SteppedLR
                override_generator=pe.natural_sgd_generator,
                algorithms=["Random",
                            "Random_SteppedLR",
                            "Lorraine",
                            "Baydin",
                            "Ours_LR",
                            "Ours_LR_Momentum",
                            "Ours_HDLR_Momentum",
                            "DiffThroughOpt",
                            "Random_Validation"],
                main_function=train.main)"""
python_cmd = textwrap.dedent(python_cmd)
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/fashion_mnist.yaml -g 'ICLR Fashion-MNIST' -l {log_root}"
COMMANDS['ICLR Fashion-MNIST'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)


# # 'ICLR CIFAR-10'
# python_cmd = """\
# import parallel_exec as pe
# import train
# pe.run_parallel(num_workers=1,
#                 num_repetitions=50,
#                 # lr_multiplier is stripped for all algorithms other than
#                 # Random_SteppedLR
#                 override_generator=pe.natural_sgd_generator,
#                 algorithms=["Random",
#                             "Random_SteppedLR",
#                             "Lorraine",
#                             "Baydin",
#                             "Ours_LR",
#                             "Ours_LR_Momentum",
#                             "Ours_HDLR_Momentum",
#                             "DiffThroughOpt",
#                             "Random_Validation"],
#                 main_function=train.main)"""
# python_cmd = textwrap.dedent(python_cmd)
# shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/cifar10.yaml -g 'ICLR CIFAR-10' -l {log_root}"
# COMMANDS['ICLR CIFAR-10'] = shell_cmd
# # subprocess.run(shell_cmd, shell=True)


# # 'ICLR PennTreebank'
# python_cmd = """\
# import parallel_exec as pe
# import train
# pe.run_parallel(num_workers=1,
#                 num_repetitions=50,
#                 # lr_multiplier is stripped for all algorithms other than
#                 # Random_SteppedLR
#                 override_generator=pe.natural_sgd_generator,
#                 algorithms=["Random",
#                             "Random_SteppedLR",
#                             "Lorraine",
#                             "Baydin",
#                             "Ours_LR",
#                             "Ours_LR_Momentum",
#                             "Ours_HDLR_Momentum",
#                             "DiffThroughOpt",
#                             "Random_Validation"],
#                 main_function=train.main)"""
# python_cmd = textwrap.dedent(python_cmd)
# shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/penn_treebank.yaml -g 'ICLR PennTreebank' -l {log_root}"
# COMMANDS['ICLR PennTreebank'] = shell_cmd
# # subprocess.run(shell_cmd, shell=True)


# 'ICLR ProofOfConcept_Revised'
python_cmd = """\
import parallel_exec as pe
import train
def natural_sgd_generator_without_momentum():
    config = pe.natural_sgd_generator(scheduled_lr=False)
    config["config_dicts"]["network_weight"].pop("momentum")
    return config
def rescaled_inits():
    config = natural_sgd_generator_without_momentum()
    # Rescale learning rate from [-6, -1] to [-6, 1]
    lr = config["config_dicts"]["network_weight"]["lr"]
    config["config_dicts"]["network_weight"]["lr"] = (lr + 1) * (7/5) + 1
    # Rescale weight_decay from [-7, -2] to [-7, 2]
    wd = config["config_dicts"]["network_weight"]["weight_decay"]
    config["config_dicts"]["network_weight"]["weight_decay"] = (wd + 2) * (9/5) + 2
    return config
pe.run_parallel(num_workers=8,
                num_repetitions=500,
                override_generator=rescaled_inits,
                algorithms=["Random"],
                main_function=train.main)
pe.run_parallel(num_workers=8,
                num_repetitions=50,
                override_generator=natural_sgd_generator_without_momentum,
                algorithms=["Ours_LR"],
                main_function=train.main)
"""
python_cmd = textwrap.dedent(python_cmd)
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml -g 'ICLR ProofOfConcept_Revised' -l {log_root}"
COMMANDS['ICLR ProofOfConcept_Revised'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)


# 'ICLR Sensitivity UCI_Energy DiffThroughOpt'
# 'ICLR Sensitivity UCI_Energy Ours_LR_Momentum'
# 'ICLR Sensitivity UCI_Energy Random'
# Start with Ours_LR_Momentum
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=8,
                num_repetitions=None,
                # lr_multiplier is stripped for all algorithms other than
                # Random_SteppedLR
                override_generator=pe.hyperparameter_ablation_initialisations,
                algorithms=["Ours_LR_Momentum"],
                main_function=train.main)"""
python_cmd = textwrap.dedent(python_cmd)
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml -g 'ICLR Sensitivity UCI_Energy Ours_LR_Momentum' -l {log_root}"
COMMANDS['ICLR Sensitivity UCI_Energy Ours_LR_Momentum'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)
# Then DiffThroughOpt, using the Ours_LR_Momentum configs
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=8,
                num_repetitions=None,
                # lr_multiplier is stripped for all algorithms other than
                # Random_SteppedLR
                override_generator=lambda: pe.diff_through_opt_configs_from_ablation_base("{directory_prefix}"),
                algorithms=["DiffThroughOpt"],
                main_function=train.main)"""
python_cmd = python_cmd.format(
    directory_prefix=os.path.join(log_root, 'ICLR Sensitivity UCI_Energy Ours_LR_Momentum'))
python_cmd = textwrap.dedent(python_cmd)
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml -g 'ICLR Sensitivity UCI_Energy DiffThroughOpt' -l {log_root}"
COMMANDS['ICLR Sensitivity UCI_Energy DiffThroughOpt'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)
# Then Random, using part of the Ours_LR_Momentum configs
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=8,
                num_repetitions=None,
                # lr_multiplier is stripped for all algorithms other than
                # Random_SteppedLR
                override_generator=lambda: pe.random_configs_from_ablation_base("{directory_prefix}"),
                algorithms=["Random"],
                main_function=train.main)"""
python_cmd = python_cmd.format(
    directory_prefix=os.path.join(log_root, 'ICLR Sensitivity UCI_Energy Ours_LR_Momentum'))
python_cmd = textwrap.dedent(python_cmd)
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml -g 'ICLR Sensitivity UCI_Energy Random' -l {log_root}"
COMMANDS['ICLR Sensitivity UCI_Energy Random'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)


# 'ICLR HypergradientComparison'
# Start with Ours_LR_Momentum
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=8,
                num_repetitions=None,
                # lr_multiplier is stripped for all algorithms other than
                # Random_SteppedLR
                override_generator=pe.hypergradient_comparison_ablation_initialisations,
                algorithms=["Ours_LR_Momentum"],
                main_function=train.main)"""
python_cmd = textwrap.dedent(python_cmd)
extra_args = "--hyperparameter_clipping '{}'"
ours_lr_momentum_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml -g 'ICLR HypergradientComparison' -l {log_root} {extra_args}"
# Then repeat with DiffThroughOpt
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=8,
                num_repetitions=None,
                # lr_multiplier is stripped for all algorithms other than
                # Random_SteppedLR
                override_generator=lambda: pe.diff_through_opt_configs_from_ablation_base("{directory_prefix}"),
                algorithms=["DiffThroughOpt"],
                main_function=train.main)"""
python_cmd = python_cmd.format(
    directory_prefix=os.path.join(log_root, 'ICLR HypergradientComparison'))
python_cmd = textwrap.dedent(python_cmd)
extra_args = "--hyperparameter_clipping '{}'"
diff_through_opt_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml -g 'ICLR HypergradientComparison' -l {log_root} {extra_args}"
COMMANDS['ICLR HypergradientComparison'] = ' && '.join((ours_lr_momentum_cmd,
                                                        diff_through_opt_cmd))
# subprocess.run(shell_cmd, shell=True)


# 'ICLR BayesOpt_fashion_mnist'
# 'ICLR BayesOpt_uci_energy'
# 'ICLR BayesOpt_uci_kin8nm'
# 'ICLR BayesOpt_uci_power'
for dataset in ('uci_energy',
                'uci_kin8nm',
                'uci_power',
                'fashion_mnist'):
    python_cmd = """\
    import parallel_exec as pe
    import bayesopt
    pe.run_parallel(num_workers=8,
                    num_repetitions=32,
                    # lr_multiplier is stripped for all algorithms other than
                    # Random_SteppedLR
                    override_generator=pe.iteration_id(),
                    algorithms=["Random_Validation_BayesOpt"],
                    main_function=bayesopt.main)"""
    python_cmd = textwrap.dedent(python_cmd)
    extra_args = "--hyperparameter_clipping '{}'"
    shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/{dataset}.yaml -g 'ICLR BayesOpt_{dataset}' -l {log_root} {extra_args}"
    COMMANDS[f'ICLR BayesOpt_{dataset}'] = shell_cmd
    # subprocess.run(shell_cmd, shell=True)


# 'ICLR Fashion-MNIST_BatchNorm'
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=8,
                num_repetitions=None,
                override_generator=lambda: pe.sgd_from_directory("{root_directory}", scheduled_lr=True),
                algorithms=["Random",
                            "Random_SteppedLR",
                            "Lorraine",
                            "Baydin",
                            "Ours_LR",
                            "Ours_LR_Momentum",
                            "Ours_HDLR_Momentum",
                            "DiffThroughOpt",
                            "Random_Validation"],
                main_function=train.main)"""
python_cmd = python_cmd.format(
    root_directory=os.path.join(log_root, 'ICLR Fashion-MNIST'))
python_cmd = textwrap.dedent(python_cmd)
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/fashion_mnist_BatchNorm.yaml -g 'ICLR Fashion-MNIST_BatchNorm' -l {log_root}"
COMMANDS['ICLR Fashion-MNIST_BatchNorm'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)


# 'ICLR uci_energy_LongDiffThroughOpt_Standalone'
# 'ICLR uci_kin8nm_LongDiffThroughOpt_Standalone'
# 'ICLR uci_power_LongDiffThroughOpt_Standalone'
for dataset in ('uci_energy',
                'uci_kin8nm',
                'uci_power'):
    python_cmd = """\
    import parallel_exec as pe
    import train
    pe.run_parallel(num_workers=6,
                    num_repetitions=60,
                    # lr_multiplier is stripped for all algorithms other than
                    # Random_SteppedLR
                    override_generator=lambda: pe.natural_sgd_generator(scheduled_lr=False),
                    algorithms=["Random",
                                "Ours_LR_Momentum",
                                "DiffThroughOpt",
                                "LongDiffThroughOpt_UCI_Standalone"],
                    main_function=train.main)"""
    python_cmd = textwrap.dedent(python_cmd)
    extra_args = "--hyperparameter_steps 20"
    shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/{dataset}.yaml -g 'ICLR {dataset}_LongDiffThroughOpt_Standalone' -l {log_root} {extra_args}"
    COMMANDS[f'ICLR {dataset}_LongDiffThroughOpt_Standalone'] = shell_cmd
    # subprocess.run(shell_cmd, shell=True)


# 'ICLR Fashion-MNIST_LongDiffThroughOpt_Standalone'
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=6,
                num_repetitions=60,
                # lr_multiplier is stripped for all algorithms other than
                # Random_SteppedLR
                override_generator=lambda: pe.natural_sgd_generator(scheduled_lr=False),
                algorithms=["Random",
                            "Ours_LR_Momentum",
                            "DiffThroughOpt",
                            "LongDiffThroughOpt_Fashion-MNIST_Standalone"],
                main_function=train.main)"""
python_cmd = textwrap.dedent(python_cmd)
extra_args = "--hyperparameter_steps 100"
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/fashion_mnist.yaml -g 'ICLR Fashion-MNIST_LongDiffThroughOpt_Standalone' -l {log_root} {extra_args}"
COMMANDS[f'ICLR Fashion-MNIST_LongDiffThroughOpt_Standalone'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)


# 'ICLR Fashion-MNIST_LongDiffThroughOpt_Medium'
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=8,
                num_repetitions=None,
                override_generator=lambda: pe.sgd_from_directory("{root_directory}"),
                algorithms=["LongDiffThroughOpt_Fashion-MNIST_Medium"],
                main_function=train.main)"""
python_cmd = python_cmd.format(
    root_directory=os.path.join(log_root, 'ICLR Fashion-MNIST'))
python_cmd = textwrap.dedent(python_cmd)
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/fashion_mnist.yaml -g 'ICLR Fashion-MNIST_LongDiffThroughOpt_Medium' -l {log_root}"
COMMANDS['ICLR Fashion-MNIST_LongDiffThroughOpt_Medium'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)


# 'ICLR uci_energy_LongDiffThroughOpt_Full'
python_cmd = """\
import parallel_exec as pe
import train
pe.run_parallel(num_workers=4,
                num_repetitions=36,
                override_generator=lambda: pe.natural_sgd_generator(scheduled_lr=False),
                algorithms=["LongDiffThroughOpt_UCI_Full"],
                main_function=train.main)"""
python_cmd = textwrap.dedent(python_cmd)
shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml -g 'ICLR uci_energy_LongDiffThroughOpt_Full' -l {log_root}"
COMMANDS['ICLR uci_energy_LongDiffThroughOpt_Full'] = shell_cmd
# subprocess.run(shell_cmd, shell=True)


# 'ICLR uci_energy_ValidationProportion0.25'
# 'ICLR uci_energy_ValidationProportion0.375'
# 'ICLR uci_energy_ValidationProportion0.5'
for validation_proportion in (0.25, 0.375, 0.5):
    python_cmd = """\
    import parallel_exec as pe
    import train
    pe.run_parallel(num_workers=8,
                    num_repetitions=200,
                    # lr_multiplier is stripped for all algorithms other than
                    # Random_SteppedLR
                    override_generator=pe.natural_sgd_generator,
                    algorithms=["Ours_LR_Momentum",
                                "Ours_HDLR_Momentum"],
                    main_function=train.main)"""
    python_cmd = textwrap.dedent(python_cmd)
    extra_args = f"--validation_proportion {validation_proportion}"
    shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml -g 'ICLR uci_energy_ValidationProportion{validation_proportion}' -l {log_root} {extra_args}"
    COMMANDS[f'ICLR uci_energy_ValidationProportion{validation_proportion}'] = shell_cmd
    # subprocess.run(shell_cmd, shell=True)


# 'ASHA Random_TrainingSetOnly'
# 'ASHA Ours_LR_Momentum'
for algorithm in ('Random_TrainingSetOnly', 'Ours_LR_Momentum'):
    python_cmd = """\
    import os
    import parallel_exec as pe
    import train
    # Python <=3.8 uses a relative __file__; force it to be absolute
    __file__ = os.path.abspath("parallel_exec.py")
    pe.ray_tune_run_asha(num_workers=8,
                         num_repetitions=114286,
                         name="{algorithm_name}",
                         local_dir="{local_dir}")
    """
    python_cmd = python_cmd.format(algorithm_name=algorithm,
                                   local_dir=log_root)
    python_cmd = textwrap.dedent(python_cmd)
    shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml ./configs/{algorithm}.yaml -g 'ASHA {algorithm}_Raw' -l {log_root}"
    COMMANDS[f'ASHA {algorithm}'] = shell_cmd
    # subprocess.run(shell_cmd, shell=True)


# 'PBT Random_TrainingSetOnly'
# 'PBT Ours_LR_Momentum'
for algorithm in ('Random_TrainingSetOnly', 'Ours_LR_Momentum'):
    python_cmd = """\
    import os
    import parallel_exec as pe
    import train
    # Python <=3.8 uses a relative __file__; force it to be absolute
    __file__ = os.path.abspath("parallel_exec.py")
    pe.ray_tune_run_pbt(num_workers=8,
                        num_repetitions=200,
                        name="{algorithm_name}",
                        local_dir="{local_dir}")
    """
    python_cmd = python_cmd.format(algorithm_name=algorithm,
                                   local_dir=log_root)
    python_cmd = textwrap.dedent(python_cmd)
    shell_cmd = f"CUDA_VISIBLE_DEVICES={cuda_device} python -c '{python_cmd}' -c ./configs/uci_energy.yaml ./configs/{algorithm}.yaml -g 'PBT {algorithm}_Raw' -l {log_root}"
    COMMANDS[f'PBT {algorithm}'] = shell_cmd
    # subprocess.run(shell_cmd, shell=True)


def _no_return_util_get_tags(*args, **kwargs):
    """Wrapper around util.get_tags which returns no value, circumventing
    issues returning large objects with multiprocessing.
    """
    # Swallow return value
    util.get_tags(*args, **kwargs)


def parse_results(num_workers=4, commands=COMMANDS.keys()):
    """Loop through the entire results sets of `commands`, constructing caches
    for each set of results."""
    non_parsed_cmds = set((
        'ASHA Random_TrainingSetOnly',
        'ASHA Ours_LR_Momentum',
        'PBT Random_TrainingSetOnly',
        'PBT Ours_LR_Momentum',
    ))
    ordered_cmds = set((
        'ICLR BayesOpt_fashion_mnist',
        'ICLR BayesOpt_uci_energy',
        'ICLR BayesOpt_uci_kin8nm',
        'ICLR BayesOpt_uci_power',
    ))
    nested_results = set((
        'ICLR BayesOpt_fashion_mnist',
        'ICLR BayesOpt_uci_energy',
        'ICLR BayesOpt_uci_kin8nm',
        'ICLR BayesOpt_uci_power',
        'ICLR HypergradientComparison',
        'ICLR Sensitivity UCI_Energy DiffThroughOpt',
        'ICLR Sensitivity UCI_Energy Ours_LR_Momentum',
    ))

    def print_complete(*args, result_name, **kwargs):
        print(f"{result_name} parse complete")

    def print_error(*args, result_name, **kwargs):
        print(f"Error parsing {result_name}")

    pool = multiprocessing.Pool(processes=num_workers)
    for folder_name in commands:
        if folder_name in non_parsed_cmds:
            continue
        folder_path = os.path.join(log_root, folder_name)
        if not os.path.exists(folder_path):
            print(f"Skipping non-existent directory {folder_name}")
            continue
        if folder_name in nested_results:
            for subfolder in os.scandir(folder_path):
                if not subfolder.is_dir():
                    continue
                pool.apply_async(
                    _no_return_util_get_tags,
                    kwds=dict(directory=os.path.join(folder_path, subfolder.name),
                              ordered=(folder_name in ordered_cmds),
                              penn_treebank=(folder_name == 'ICLR PennTreebank')),
                    callback=partial(print_complete, result_name=f"{folder_name}/{subfolder.name}"),
                    error_callback=partial(print_error, result_name=f"{folder_name}/{subfolder.name}"))
        else:
            pool.apply_async(
                _no_return_util_get_tags,
                kwds=dict(directory=folder_path,
                          ordered=(folder_name in ordered_cmds),
                          penn_treebank=(folder_name == 'ICLR PennTreebank')),
                callback=partial(print_complete, result_name=folder_name),
                error_callback=partial(print_error, result_name=folder_name))
    pool.close()
    pool.join()


def run(commands=COMMANDS.keys()):
    """Execute `commands`."""
    if not os.path.exists(log_root):
        os.makedirs(log_root)
    logging.basicConfig(filename=os.path.join(log_root, 'run.log'),
                        level=logging.DEBUG,
                        # force=True,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    for command in commands:
        logging.info(f"Starting command {command}")
        result = subprocess.run(COMMANDS[command],
                                shell=True,
                                capture_output=False)
        result_logger = logging.info
        if result.returncode != 0:
            result_logger = logging.error
            if result.stdout:
                stdout_path = os.path.join(log_root, f'{command}.stdout.txt')
                with open(stdout_path, 'wb') as stdout_file:
                    stdout_file.write(result.stdout)
            if result.stderr:
                stderr_path = os.path.join(log_root, f'{command}.stderr.txt')
                with open(stderr_path, 'wb') as stderr_file:
                    stderr_file.write(result.stderr)
        result_logger(f"Command {command} terminated "
                      f"with return code {result.returncode}")
        if os.path.exists(os.path.join(log_root, 'STOP')):
            logging.info("STOP file exists; terminating runner now")
            break


if __name__ == '__main__':
    run(commands=COMMANDS)

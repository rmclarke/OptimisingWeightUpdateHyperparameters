This repository contains code for the paper _Scalable One-Pass Optimisation of
High-Dimensional Weight-Update Hyperparameters by Implicit Differentiation_.

# Installation
Our dependencies are fully specified in `Pipfile`, which can be supplied to
`pipenv` to install the environment. One failsafe approach is to install
`pipenv` in a fresh virtual environment, then run `pipenv install` in this
directory. Note the Pipfile specifies our Python 3.9 development environment;
most experiments were run in an identical environment under Python 3.7 instead.

Difficulties with CUDA versions meant we had to manually install PyTorch and
Torchvision rather than use `pipenv` --- the corresponding lines in `Pipfile`
may need adjustment for your use case. Alternatively, use the list of
dependencies as a guide to what to install yourself with `pip`, or use the full
dump of our development environment in `final_requirements.txt`.

Datasets may not be bundled with the repository, but are expected to be found at
locations specified in `datasets.py`, preprocessed into single PyTorch tensors
of all the input and output data (generally `data/<dataset>/data.pt` and
`data/<dataset>/targets.pt`). 

# Configuration
Training code is controlled with YAML configuration files, as per the examples
in `configs/`. Generally one file is required to specify the dataset, and a
second to specify the algorithm, using the obvious naming convention. Brief help
text is available on the command line, but the meanings of each option should be
reasonably self-explanatory.

For _Ours (WD+LR)_, use the file `Ours_LR.yaml`; for _Ours (WD+LR+M)_, use the
file `Ours_LR_Momentum.yaml`; for _Ours (WD+HDLR+M)_, use the file
`Ours_HDLR_Momentum.yaml`. For _Long/Medium/Full Diff-through-Opt_, we provide
separate configuration files for the UCI cases and the Fashion-MNIST cases.

We provide two additional helper configurations. `Random_Validation.yaml` copies
`Random.yaml`, but uses the entire validation set to compute the validation loss
at each logging step. This allows for stricter analysis of the best-performing
run at particular time steps, for instance while constructing _Random (3-batched)_.
`Random_Validation_BayesOpt.yaml` only forces the use of the entire dataset for
the very last validation loss computation, so that Bayesian Optimisation runs
can access reliable performance metrics without adversely affecting runtime.

The configurations provided match those necessary to replicate the main
experiments in our paper (in Section 4: Experiments). Other trials, such as
those in the Appendix, will require these configurations to be modified as we
describe in the paper. Note especially that our three short-horizon bias studies
all require different modifications to the `LongDiffThroughOpt_*.yaml`
configurations. 

# Running
Individual runs are commenced by executing `train.py` and passing the desired
configuration files with the `-c` flag. For example, to run the default Fashion-MNIST experiments
using Diff-through-Opt, use:
```shell
$ python train.py -c ./configs/fashion_mnist.yaml ./configs/DiffThroughOpt.yaml
```

Bayesian Optimisation runs are started in a similar way, but with a call to
`bayesopt.py` rather than `train.py`.

For executing multiple runs in parallel, `parallel_exec.py` may be useful:
modify the main function call at the bottom of the file as required, then
call this file instead of `train.py` at the command line. The number of parallel
workers may be specified by `num_workers`. Any configurations passed at the command line are used
as a base, to which modifications may be added by `override_generator`. The
latter should either be a function which generates one override dictionary per
call (in which case `num_repetitions` sets the number of overrides to generate),
or a function which returns a generator over configurations (in which case set
`num_repetitions = None`). Each configuration override is run once for each of
`algorithms`, whose configurations are read automatically from the corresponding
files and should not be explicitly passed at the command line. Finally,
`main_function` may be used to switch between parallel calls to `train.py` and
`bayesopt.py` as required.

For blank-slate replications, the most useful override generators will be
`natural_sgd_generator`, which generates a full SGD initialisation in the ranges
we use, and `iteration_id`, which should be used with Bayesian Optimisation runs
to name each parallel run using a counter. Other generators may be useful if you
wish to supplement existing results with additional algorithms etc.

PennTreebank and CIFAR-10 were executed on clusters running SLURM; the
corresponding subfolders contain configuration scripts for these experiments,
and `submit.sh` handles the actual job submission.

# Analysis
By default, runs are logged in Tensorboard format to the `./runs` directory,
where Tensorboard may be used to inspect the results. If desired, a descriptive
name can be appended to a particular execution using the `-n` switch on the
command line. Runs can optionally be written to a dedicated subfolder specified
with the `-g` switch, and the base folder for logging can be changed with the
`-l` switch.

If more precise analysis is desired, pass the directory containing the desired
results to `util.get_tags()`, which will return a dictionary of the evolution of
each logged scalar in the results. Note that this function uses Tensorboard calls
which predate its `--load_fast` option, so may take tens of minutes to return.

This data dictionary can be passed to one of the more involved plotting routines
in `figures.py` to produce specific plots. The script `paper_plots.py` generates
all the plots we use in our paper, and may be inspected for details of any
particular plot.

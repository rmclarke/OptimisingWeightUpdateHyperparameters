"""Script and utilities for performing Bayesian Optimisation over
hyperparameters.
"""
import copy
import os
from functools import partial

import bayes_opt as bo
import numpy as np
import torch as to

import train
import util


def launch_run(config_dict, weight_decay, lr, momentum):
    """Start a single run with the given hyperparameters."""
    momentum = -np.log((1 / momentum) - 1)
    config_override = {
        'config_dicts': {
            'network_weight': {
                'weight_decay': weight_decay,
                'lr': lr,
                'momentum': momentum
            }}}
    dataset = config_dict['config_dicts']['dataset']['class']
    # Stop train.main stripping out bits of our config
    config_dict = copy.deepcopy(config_dict)
    (final_validation_loss,
     final_test_loss) = train.main(config_dict=config_dict,
                                   config_override=config_override)
    if not np.isfinite(final_validation_loss):
        # Approximate upper bound on initial losses
        final_validation_loss = (
            to.tensor(150) if dataset == 'UCI_Energy' else
            to.tensor(0.1) if dataset == 'UCI_Kin8nm' else
            to.tensor(400) if dataset == 'UCI_Power' else
            to.tensor(3) if dataset == 'FashionMNIST' else
            to.tensor(float('nan')))
    return -final_validation_loss


def main(config_dict=None, config_override={}):
    """Main entry point for Bayesian Optimisation."""
    log_folder = config_override.pop('log_folder', '.')
    util.nested_update(config_dict, config_override)

    bounds = {'lr': (-6, -1),
              'weight_decay': (-7, -2),
              # Avoid impossible sigmoidal transforms
              'momentum': (0 + 1e-6, 1 - 1e-6)}
    optimiser = bo.BayesianOptimization(
        f=partial(launch_run, config_dict=config_dict),
        pbounds=bounds)

    logger = bo.logger.JSONLogger(
        path=os.path.join(log_folder, 'log.json'))
    optimiser.subscribe(bo.event.Events.OPTIMIZATION_START, logger)
    optimiser.subscribe(bo.event.Events.OPTIMIZATION_STEP, logger)
    optimiser.subscribe(bo.event.Events.OPTIMIZATION_END, logger)

    # Using default arguments
    optimiser.maximize(init_points=5,
                       n_iter=25)


if __name__ == '__main__':
    main()

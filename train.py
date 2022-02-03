"""Main system training function and helpers."""

import copy
import json
import os
import pathlib
from contextlib import contextmanager
from itertools import repeat, chain

import higher
import numpy as np
import ray.tune as tune
import torch as to
import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
import datasets
import models
import optimisers
import util


class Learner():
    def __init__(self,
                 device,
                 batch_size,
                 network_weight_steps,
                 total_step_limit,
                 hyperparameter_steps,
                 validation_proportion,
                 optimised_hyperparameters,
                 transformed_hyperparameters,
                 high_dimensional_hyperparameters,
                 config_dicts,
                 tensorboard_tracker,
                 network_weight_grad_clipping,
                 hyperparameter_clipping,
                 reset_model_after_hyperparameter_update,
                 renew_model_reset_state_interval,
                 patch_optimiser,
                 null_zero_validation_datasets,
                 force_single_network_weight_step,
                 multi_batch_test_dataset_evaluations,
                 penn_treebank_validation_override,
                 full_batch_validation_evaluations,
                 reset_loop_before_hyperparameter_step,
                 process_id=0,
                 _ray_tune_config=False):

        self.config_dicts = config_dicts
        self.device = device
        self.batch_size = batch_size
        self.network_weight_steps = network_weight_steps
        self.hyperparameter_steps = hyperparameter_steps
        self.total_step_limit = total_step_limit
        self.validation_proportion = validation_proportion
        self.optimised_hyperparameters = optimised_hyperparameters
        self.transformed_hyperparameters = transformed_hyperparameters
        self.high_dimensional_hyperparameters = (
            high_dimensional_hyperparameters)
        self.network_weight_grad_clipping = network_weight_grad_clipping
        self.hyperparameter_clipping = hyperparameter_clipping
        self.reset_loop_before_hyperparameter_step = reset_loop_before_hyperparameter_step
        self.reset_network_weight_grads = False
        self.reset_model_after_hyperparameter_update = (
            reset_model_after_hyperparameter_update)
        self.renew_model_reset_state_interval = renew_model_reset_state_interval
        self.patch_optimiser = patch_optimiser
        self.null_zero_validation_datasets = null_zero_validation_datasets
        self.force_single_network_weight_step = force_single_network_weight_step
        self.multi_batch_test_dataset_evaluations = multi_batch_test_dataset_evaluations
        self.full_batch_validation_evaluations = full_batch_validation_evaluations
        self.penn_treebank_validation_override = penn_treebank_validation_override

        if self.force_single_network_weight_step:
            self.hyperparameter_steps *= self.network_weight_steps
            self.network_weight_steps = 1

        self.accuracies = {}
        self.perplexities = {}
        self.unnormalised_losses = {}
        self.hidden_states = {'Training': [],
                              'Validation': [],
                              'Test': []}

        self.network_weight_step = 0
        self.hyperparameter_step = 0
        self.inv_hessian_saved = False
        self.last_hypergradients = {}

        self.tracker = tensorboard_tracker
        self.process_id = process_id
        self._ray_tune_config = _ray_tune_config

        self.model = getattr(models,
                             self.config_dicts['model']['class'])(
            **config.get_args(self.config_dicts['model']))
        self.loss_function = getattr(models,
                                     self.config_dicts['loss']['class'])(
            **config.get_args(self.config_dicts['loss']))

        self.check_config_validity()
        self.enable_hyperparameter_grads()
        self.enable_hyperparameter_high_dimensionality()
        self.construct_dataloaders()
        self.construct_optimisers()

        # Needed so we can detect Ray Tune perturbations
        self.initial_network_weight_dict = copy.deepcopy(
            self.config_dicts['network_weight'])

    def check_config_validity(self):
        """Make sure the requested combination of configuration settings makes
        sense.
        """
        if 'hyperparameter' in self.config_dicts:
            if hasattr(to.optim,
                       self.config_dicts['network_weight']['class']):
                assert self.patch_optimiser
            else:
                optimiser_class = getattr(optimisers, self.config_dicts['network_weight']['class'])
                if getattr(optimiser_class, 'patch_optimiser', False):
                    assert self.patch_optimiser

    def enable_hyperparameter_grads(self):
        """Replace all hyperparameters which will be optimised with
        requires_grad versions.
        """
        for key in self.optimised_hyperparameters:
            if key == 'curvature_transform':
                self.config_dicts['network_weight'][key]['diagonal_value'] = to.tensor(
                    self.config_dicts['network_weight'][key]['diagonal_value'],
                    requires_grad=True,
                    device=self.device)
            else:
                self.config_dicts['network_weight'][key] = to.tensor(
                    self.config_dicts['network_weight'][key],
                    requires_grad=True,
                    device=self.device)

    def enable_hyperparameter_high_dimensionality(self):
        for key in config.get_args(self.config_dicts['network_weight']):
            if key in self.high_dimensional_hyperparameters:
                existing_hyperparameter = (
                    self.config_dicts['network_weight'][key])
                self.config_dicts['network_weight'][key] = []
                for parameter in self.model.parameters():
                    template = to.ones_like(parameter, device=self.device)
                    if key == 'curvature_transform':
                        new_hyperparameter = template * existing_hyperparameter['diagonal_value']
                        cholesky = to.diagflat(new_hyperparameter.sqrt().view(-1))
                        if existing_hyperparameter.get('noise_variance', 0):
                            cholesky += (to.randn_like(cholesky)
                                        * existing_hyperparameter['noise_variance'])
                        new_hyperparameter = cholesky.tril()
                    else:
                        new_hyperparameter = template * existing_hyperparameter
                    self.config_dicts['network_weight'][key].append(
                        new_hyperparameter
                        .detach().requires_grad_(True))
            else:
                self.config_dicts['network_weight'][key] = repeat(
                    self.config_dicts['network_weight'][key])

    def construct_dataloaders(self):
        """Construct training, validation and test dataloaders for the dataset.
        """
        dataset_class = getattr(datasets, self.config_dicts['dataset']['class'])
        (training_dataset,
         validation_dataset,
         test_dataset) = datasets.make_split_datasets(
             dataset_class,
             validation_proportion=self.validation_proportion,
             **config.get_args(self.config_dicts['dataset']))

        self.track_accuracies = getattr(
            dataset_class, 'track_accuracies', False)
        self.track_perplexities = getattr(
            dataset_class, 'track_perplexities', False)
        has_target_data = getattr(
            dataset_class, 'has_target_data')

        self.training_dataloader = to.utils.data.DataLoader(
            training_dataset,
            batch_size=self.batch_size,
            shuffle=has_target_data)

        if self.full_batch_validation_evaluations and not self.penn_treebank_validation_override:
            self.validation_dataloader = repeat(validation_dataset[:])
            self.full_validation_dataloader = self.validation_dataloader
        else:
            self.full_validation_dataloader = repeat(validation_dataset[:])
            self.validation_dataloader = to.utils.data.DataLoader(
                validation_dataset,
                batch_size=self.batch_size,
                shuffle=has_target_data)

        if self.multi_batch_test_dataset_evaluations:
            self.test_dataloader = to.utils.data.DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                shuffle=False)
        else:
            self.test_dataloader = repeat(test_dataset[:])

        if (self.validation_proportion == 0 and not self.null_zero_validation_datasets):
            self.validation_dataloader = self.training_dataloader


    def construct_optimisers(self):
        """Construct the top-level optimiser for network weights (and possibly
        hyperparameters).
        """
        self.lr_multiplier = self.config_dicts['network_weight'].pop(
            'lr_multiplier', None)
        if isinstance(self.lr_multiplier, repeat):
            self.lr_multiplier = next(self.lr_multiplier)

        # Initialise network_weight_optimiser with any value of learning rate,
        # then immediately overwrite it
        self.network_weight_optimiser = getattr(
            optimisers,
            self.config_dicts['network_weight']['class'])(
                [{'params': p} for p in self.model.parameters()],
                lr=1)
        for hyperparameter_key, hyperparameter_values in config.get_args(
                self.config_dicts['network_weight']).items():
            for param_group, hyperparameter in zip(
                    self.network_weight_optimiser.param_groups,
                    hyperparameter_values):
                param_group[hyperparameter_key] = hyperparameter

        if 'hyperparameter' not in self.config_dicts:
            # We aren't creating a hyperparameter optimiser, so the
            # network_weight optimiser will be a PyTorch instance rather than a
            # Higher instance, so make sure the rest of our code is actually
            # compatible with the former
            assert not self.optimised_hyperparameters
            self.hyperparameter_optimiser = None
            self.reset_network_weight_grads = True
            # Default PyTorch optimisers expect callable closures
            original_network_weight_optimiser_step = self.network_weight_optimiser.step

            def new_network_weight_step(training_loss):
                training_loss.backward()
                self.network_weight_grad_callback(
                    all_params=list(
                        chain(
                            *(group['params']
                              for group in self.network_weight_optimiser.param_groups))))
                return original_network_weight_optimiser_step(
                    lambda: training_loss)
            self.network_weight_optimiser.step = new_network_weight_step
            return

        hyperparameters = []
        for key in self.optimised_hyperparameters:
            hyperparameter = self.config_dicts['network_weight'][key]
            if key in self.high_dimensional_hyperparameters:
                # High-dimensional hyperparameters come in lists, so we can
                # directly extend here
                hyperparameters.extend(hyperparameter)
            else:
                # Non-high-dimensional hyperparameters come in repeat()
                # iterators, so take one object and append
                hyperparameters.append(next(hyperparameter))
        hyperparameter_base_optimiser = getattr(
            optimisers,
            self.config_dicts['hyperparameter']['class'])(
                hyperparameters,
                **config.get_args(self.config_dicts['hyperparameter']))
        self.hyperparameter_optimiser = getattr(
            optimisers,
            self.config_dicts['hyperparameter_wrapper']['class'])(
                hyperparameter_base_optimiser,
                **config.get_args(self.config_dicts['hyperparameter_wrapper']))
        if hasattr(self.hyperparameter_optimiser,
                   'max_useful_parameter_age'):
            if (self.reset_loop_before_hyperparameter_step == -1
                or self.hyperparameter_optimiser.max_useful_parameter_age
                    < self.reset_loop_before_hyperparameter_step):
                self.reset_loop_before_hyperparameter_step = self.hyperparameter_optimiser.max_useful_parameter_age

    def network_weight_grad_callback(self, all_params=None, all_grads=None):
        """Apply any network_weight gradient transformations required, such as
        clipping.
        """
        # all_params is used by normal PyTorch optimisers, which clip in-place
        # all_grads is used by Higher optimisers, which clip out of place
        if 'norm' in self.network_weight_grad_clipping:
            if all_params:
                to.nn.utils.clip_grad_norm_(
                    all_params, max_norm=self.network_weight_grad_clipping['norm'])
            if all_grads:
                full_norm = to.norm(
                    to.cat(
                        [grad.view(-1) for grad in all_grads]))
                clip_factor = self.network_weight_grad_clipping['norm'] / (full_norm + 1e-6)
                if clip_factor < 1:
                    all_grads = [grad * clip_factor for grad in all_grads]
                return all_grads
        else:
            if all_params:
                return all_params
            if all_grads:
                return all_grads

    @contextmanager
    def patched_modules(self, track_higher_grads):
        """Temporarily overwrite the model and network weight optimiser with
        differentiable versions from the higher library.
        """
        if not self.patch_optimiser:
            yield
            return

        with higher.innerloop_ctx(
                self.model,
                self.network_weight_optimiser,
                track_higher_grads=track_higher_grads) as \
                (functional_model, differentiable_optimiser):
            original_model = self.model
            self.model = functional_model
            self.hyperparameter_optimiser.functional_model = functional_model

            original_optimiser = self.network_weight_optimiser
            self.network_weight_optimiser = differentiable_optimiser
            for new_param_group, old_param_group in zip(
                    differentiable_optimiser.param_groups,
                    original_optimiser.param_groups):
                new_param_group.update({k: v
                                        for k, v in old_param_group.items()
                                        if k != 'params'})
            differentiable_optimiser._grad_callback = lambda all_grads: self.network_weight_grad_callback(all_grads=all_grads)

            yield

            original_model.load_state_dict(
                self.model.state_dict())
            for param, param_state in zip(
                    chain(*(g['params'] for g in original_optimiser.param_groups)),
                    chain(*(g.values() for g in self.network_weight_optimiser.state))):
                # ??? No need to detach; Higher will do that for us next time
                # we patch the optimiser
                param_state['momentum_buffer'].detach_()
                original_optimiser.state[param].update(param_state)

            self.model = original_model
            self.network_weight_optimiser = original_optimiser
            self.hyperparameter_optimiser.functional_model = None
            for hidden_key, hidden_values in self.hidden_states.items():
                self.hidden_states[hidden_key] = [d.detach()
                                                  for d in hidden_values]

    def install_hyperparameters(self):
        """Where we use transformed hyperparameters, install the corresponding
        transformation of these into the network weight optimiser for the
        current step of training.
        """
        for key, transform in self.transformed_hyperparameters.items():
            # Don't disrupt any lr_multiplier efforts if we're using them
            if (self.lr_multiplier and key == 'lr' and (
                    self.hyperparameter_step > 0
                    or self.network_weight_step > 0)):
                continue
            # Use Numpy instead of PyTorch if we aren't optimising the
            # hyperparameter, as it then won't be a Tensor
            if key in self.optimised_hyperparameters:
                lib = to
            else:
                lib = np

            if transform == 'exp':
                transform_func = lib.exp
            elif transform == '10^':
                transform_func = lambda x: 10**x
            elif transform == 'sigmoid':
                transform_func = lambda x: 1 / (1 + lib.exp(-x))
            elif transform is None:
                transform_func = lambda x: x
            for param_group, hyperparameter_schedule in zip(
                    self.network_weight_optimiser.param_groups,
                    self.config_dicts['network_weight'][key]):
                param_group[key] = transform_func(hyperparameter_schedule)

    def log_now(self, end_of_pass=False):
        """Write the current metrics to the tracker.
        """
        key_prefix = ''
        if end_of_pass:
            key_prefix = 'Last_'
            step = self.hyperparameter_step
        else:
            step = ((self.hyperparameter_step * self.network_weight_steps)
                    + self.network_weight_step)

        if self._ray_tune_config:
            # Undo Ray's directory changing so our relative paths work
            os.chdir(
                pathlib.Path(__file__).parent.resolve())

        self.tracker.add_scalar('{}Loss/Training'.format(key_prefix),
                                self.training_loss.item(),
                                step)
        self.tracker.add_scalar('{}Loss/Validation'.format(key_prefix),
                                self.validation_loss.item(),
                                step)
        self.tracker.add_scalar('{}Loss/Test'.format(key_prefix),
                                self.test_loss.item(),
                                step)
        if not end_of_pass:
            self.progress.set_postfix({
                'Loss/Training': self.training_loss.item()
            })

        for accuracy_key, accuracy_value in self.accuracies.items():
            self.tracker.add_scalar(
                '{}Accuracy/{}'.format(key_prefix, accuracy_key),
                accuracy_value.item(),
                step)
        for perplexity_key, perplexity_value in self.perplexities.items():
            self.tracker.add_scalar(
                '{}Perplexity/{}'.format(key_prefix, perplexity_key),
                perplexity_value.item(),
                step)
        for loss_key, loss_value in self.unnormalised_losses.items():
            self.tracker.add_scalar(
                '{}Unnormalised_Loss/{}'.format(key_prefix, loss_key),
                loss_value.item(),
                step)
        loggable_hyperparameters = self.optimised_hyperparameters.copy()
        if self.lr_multiplier and 'lr' not in loggable_hyperparameters:
            loggable_hyperparameters.append('lr')
        for hyperparameter_key in loggable_hyperparameters:
            if hyperparameter_key in self.high_dimensional_hyperparameters:
                self.tracker.add_histogram(
                    '{}Hyperparameter/{}'.format(key_prefix,
                                                 hyperparameter_key.title()),
                    to.cat([param_group[hyperparameter_key].view(-1)
                            for param_group
                            in self.network_weight_optimiser.param_groups]),
                    step)
            else:
                value = self.network_weight_optimiser.param_groups[0][hyperparameter_key]
                if hyperparameter_key in self.optimised_hyperparameters:
                    value = value.item()
                self.tracker.add_scalar(
                    '{}Hyperparameter/{}'.format(key_prefix,
                                                 hyperparameter_key.title()),
                    value,
                    step)
        if False:  #'curvature_transform' in self.optimised_hyperparameters:
            curvature_transform = self.network_weight_optimiser.param_groups[0]['curvature_transform']
            if self.model.weights.view(-1).shape[0] > 2:
                self.tracker.add_histogram(
                    'Second_Order/LT_Curvature_Transform',
                    curvature_transform.view(-1),
                    step)
                self.tracker.add_histogram(
                    'Weights',
                    self.model.weights,
                    step)
                self.tracker.add_pytorch_tensor(
                    f'Weights/Weights_{step}.pt',
                    self.model.weights)
                if not self.inv_hessian_saved:
                    self.tracker.add_pytorch_tensor(
                        'True_Inverse_Hessian.pt',
                        self.model.full_curvature)
                    self.inv_hessian_saved = True
            else:
                for index, weight in enumerate(self.model.weights):
                    self.tracker.add_scalar(
                        f'Weights/{index}',
                        weight,
                        step)
                for row_idx, row_data in enumerate(curvature_transform):
                    for column_idx, column_data in enumerate(row_data[:row_idx+1]):
                        self.tracker.add_scalar(
                            f'LT_Curvature_Transform/{row_idx},{column_idx}',
                            column_data,
                            step)
                        if 'curvature_transform' in self.last_hypergradients:
                            self.tracker.add_scalar(
                                f'LT_Curvature_Transform_Grad/{row_idx},{column_idx}',
                                self.last_hypergradients['curvature_transform'][row_idx, column_idx],
                                step)
                self.last_hypergradients = {
                    name: self.config_dicts['network_weight'][name][0].grad.clone()
                    for name in self.optimised_hyperparameters}
                if not self.inv_hessian_saved:
                    for row_idx, row_data in enumerate(self.model.full_curvature):
                        for column_idx, column_data in enumerate(row_data):
                            self.tracker.add_scalar(
                                f'True_Inverse_Hessian/{row_idx},{column_idx}',
                                column_data,
                                step)
                    self.inv_hessian_saved = True

    def postprocess_hyperparameters(self):
        """Perform postprocessing operations on hyperparameters after they've
        been updated by the optimiser.
        """
        # Hyperparameter clipping
        for (hyperparameter_key,
             (min_clip, max_clip)) in self.hyperparameter_clipping.items():
            if hyperparameter_key not in self.optimised_hyperparameters:
                for param_group in self.network_weight_optimiser.param_groups:
                    trial_value = self.config_dicts['network_weight'][hyperparameter_key]
                    if isinstance(trial_value, repeat):
                        # Make trial_value non-scalar so we can use any() below
                        trial_value = to.tensor([next(trial_value)])
                    if (any(trial_value < min_clip)
                            or any(trial_value > max_clip)):
                        raise ValueError(
                            f'Non-optimised hyperparameter {hyperparameter_key}'
                            ' is outside requested clipping range')
                continue

            for param_group in self.network_weight_optimiser.param_groups:
                param_group[hyperparameter_key].data.clamp_(min=min_clip,
                                                            max=max_clip)

    def train(self):
        """Execute a training run.
        """
        self.model.to(device=self.device)
        original_state_dict = copy.deepcopy(self.model.state_dict())

        repeating_training_dataloader = datasets.repeating_dataloader(
            self.training_dataloader,
            reset_callback=lambda: self.hidden_states['Training'].clear())
        repeating_validation_dataloader = datasets.repeating_dataloader(
            self.validation_dataloader,
            reset_callback=lambda: self.hidden_states['Validation'].clear())
        repeating_test_dataloader = datasets.repeating_dataloader(
            self.test_dataloader,
            reset_callback=lambda: self.hidden_states['Test'].clear())

        with tqdm.tqdm(range(self.hyperparameter_steps),
                       position=self.process_id,
                       desc=str(self.process_id)) as self.progress:
            for self.hyperparameter_step in self.progress:
                # Reset self.network_weight_step from previous loop
                self.network_weight_step = 0
                if self.hyperparameter_optimiser:
                    self.hyperparameter_optimiser.zero_grad()
                # Mechanism to allow us to reset gradients for
                # limited lookback windows in DiffThroughOpt,
                # if necessary
                continue_inner_loop = True
                inner_loop_range = range(self.network_weight_steps)
                while continue_inner_loop:
                    continue_inner_loop = False
                    track_higher_grads = True
                    if (self.reset_loop_before_hyperparameter_step != -1
                        and self.network_weight_steps - self.network_weight_step - 1
                            > self.reset_loop_before_hyperparameter_step):
                        track_higher_grads = False
                    with self.patched_modules(track_higher_grads):
                        for (self.network_weight_step,
                             training_batch,
                             validation_batch,
                             test_batch) in zip(
                                 inner_loop_range,
                                 repeating_training_dataloader,
                                 repeating_validation_dataloader,
                                 repeating_test_dataloader):
                            if ((self.hyperparameter_step * self.network_weight_steps)
                                    + self.network_weight_step == self.total_step_limit):
                                return

                            if self.reset_network_weight_grads:
                                self.network_weight_optimiser.zero_grad()
                            self.model.train()
                            self.install_hyperparameters()

                            self.training_loss = self._compute_loss(
                                training_batch, "Training")
                            self.validation_loss = self._compute_loss(
                                validation_batch, "Validation")

                            self.model.eval()
                            with to.no_grad():
                                self.test_loss = self._compute_loss(test_batch, "Test")

                            self.log_now()
                            self.network_weight_optimiser.step(self.training_loss)

                            if self._ray_tune_config and (
                                    self.network_weight_steps - self.network_weight_step > 1):
                                # This might be our last chance to report, so
                                # can't rely on the next report picking up
                                # newer values, so we refresh them here
                                with to.no_grad():
                                    new_validation_loss = self._compute_loss(
                                        validation_batch, "Validation")
                                    new_test_loss = self._compute_loss(
                                        test_batch, "Test")
                                # self._compute_loss() also updates
                                # self.unnormalised_losses
                                tune.report(validation_loss=new_validation_loss.item() if not to.isnan(new_validation_loss) else float('inf'),
                                            test_loss=new_test_loss.item(),
                                            unnormalised_test_loss=self.unnormalised_losses['Test'].item())
                            if (self.network_weight_steps - self.network_weight_step - 1
                                    == self.reset_loop_before_hyperparameter_step):
                                continue_inner_loop = True
                                inner_loop_range = range(self.network_weight_step + 1,
                                                         self.network_weight_steps)
                                break
                        if continue_inner_loop:
                            continue

                        self.log_now(end_of_pass=True)
                        if self.hyperparameter_optimiser:
                            self.model.train()
                            new_validation_loss = self._compute_loss(
                                validation_batch, "Validation")
                            # BaydinHypergradientOptimiser needs old and new
                            # losses here.
                            # All others will ignore self.training_loss
                            self.hyperparameter_optimiser.step(
                                self.training_loss, new_validation_loss)
                        if self.lr_multiplier:
                            # Need to manually add hyperparameter clipping support,
                            # because these hyperparameters aren't transformed,
                            # so we need to transform the clipping limits
                            assert self.transformed_hyperparameters['lr'] == '10^'
                            for group in self.network_weight_optimiser.param_groups:
                                group['lr'] *= self.lr_multiplier
                                if 'lr' in self.hyperparameter_clipping:
                                    group['lr'] = np.clip(
                                        group['lr'],
                                        a_min=10**self.hyperparameter_clipping['lr'][0],
                                        a_max=10**self.hyperparameter_clipping['lr'][1])
                        if self.reset_model_after_hyperparameter_update:
                            if (self.renew_model_reset_state_interval is not None
                                and ((self.hyperparameter_step + 1)
                                     % self.renew_model_reset_state_interval == 0)):
                                original_state_dict = copy.deepcopy(self.model.state_dict())
                            elif self.hyperparameter_step < self.hyperparameter_steps - 1:
                                # Only reset the model if we aren't on the very
                                # last hyperparameter_step.
                                # We'll miss logging the final pre-reset
                                # losses, but we would only be reporting these
                                # graphically, where the difference would be
                                # invisible, so the plotting error is negligible.
                                self.model.load_state_dict(original_state_dict)
                    # Outside model patch
                    if self._ray_tune_config:
                        # Checkpoint before reporting results, as the latter
                        # might cancel this run immediately
                        step = ((self.hyperparameter_step * self.network_weight_steps)
                                + self.network_weight_step)
                        with tune.checkpoint_dir(step=step) as checkpoint_dir:
                            self.save_training_state(
                                os.path.join(checkpoint_dir, 'checkpoint.pt'))
                        # This might be our last chance to report, so can't
                        # rely on the next report picking up newer values, so
                        # we refresh them here
                        with to.no_grad():
                            # It isn't unfair to recalculate
                            # new_validation_loss if we already did above,
                            # because this one is a constant overhead for
                            # all Ray configurations
                            new_validation_loss = self._compute_loss(
                                validation_batch, "Validation")
                            new_test_loss = self._compute_loss(
                                test_batch, "Test")
                        # self._compute_loss() also updates
                        # self.unnormalised_losses
                        tune.report(validation_loss=new_validation_loss.item() if not to.isnan(new_validation_loss) else float('inf'),
                                    test_loss=new_test_loss.item(),
                                    unnormalised_test_loss=self.unnormalised_losses['Test'].item())
                self.postprocess_hyperparameters()

            # Final loss evaluation for completeness of logs
            with to.no_grad(), self.patched_modules(track_higher_grads=False):
                self.model.train()
                self.install_hyperparameters()
                self.training_loss = self._compute_loss(
                    next(repeating_training_dataloader), "Training")
                self.validation_loss = self._compute_loss(
                    next(repeating_validation_dataloader), "Validation")

                self.model.eval()
                self.test_loss = self._compute_loss(
                    next(repeating_test_dataloader), "Test")

                # Log a new entry, but then put network_weight_step back where
                # it's expected
                self.network_weight_step += 1
                self.log_now()
                self.network_weight_step -= 1

    def _compute_loss(self, batch, label):
        """Perform a forward pass of the current model using `batch`, computing
        the associated loss.
        """
        if (label == 'Validation'
                and self.validation_proportion == 0
                and self.null_zero_validation_datasets):
            return to.tensor(float('nan'))

        if label == 'Test' and self.multi_batch_test_dataset_evaluations:
            loss = util.compute_full_dataset_loss(self.test_dataloader,
                                                  self.model,
                                                  self.loss_function)
        elif (label == 'Validation'
              and self.full_batch_validation_evaluations
              and self.penn_treebank_validation_override):
            loss = util.compute_full_dataset_loss(self.validation_dataloader,
                                                  self.model,
                                                  self.loss_function)
        else:
            batch_inputs, batch_targets = [b.to(self.device)
                                           for b in batch]
            predictions = self.model(batch_inputs, *self.hidden_states[label])
            if isinstance(predictions, tuple):
                predictions, self.hidden_states[label] = predictions
                self.hidden_states[label] = [
                    state.detach() for state in self.hidden_states[label]]

            if self.track_accuracies:
                self.accuracies[label] = (
                    to.argmax(predictions, dim=-1) == batch_targets).float().mean()
            if self.config_dicts['dataset'].get('normalise_outputs', False):
                assert not self.multi_batch_test_dataset_evaluations
                unnormaliser = self.training_dataloader.dataset.dataset.target_unnormaliser
                self.unnormalised_losses[label] = self.loss_function(
                    unnormaliser(predictions),
                    unnormaliser(batch_targets))
            loss = self.loss_function(predictions, batch_targets)

        if self.track_perplexities:
            assert isinstance(self.loss_function, to.nn.CrossEntropyLoss)
            self.perplexities[label] = to.exp(loss).detach()
        return loss

    def save_training_state(self, save_file, **state_updates):
        """Save the current state of training to `save_file`, such that it can
        be reinstated at a later time by initialising a similarly-configured
        Learner object and calling `load_training_state()` on it.
        """
        state_dict = {
            'model_state_dict': self.model.state_dict(),
            'initial_network_weight_dict': self.initial_network_weight_dict,
            'network_weight_dict': self.config_dicts['network_weight'],
            'network_weight_optimiser_state_dict': self.network_weight_optimiser.state_dict(),
            'network_weight_step': self.network_weight_step,
            'hyperparameter_step': self.hyperparameter_step,
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'test_loss': self.test_loss,
            'log_directory': self.tracker.log_dir}
        if self.hyperparameter_optimiser:
            state_dict['hyperparameter_optimiser_state_dict'] = (
                self.hyperparameter_optimiser.state_dict())
        state_dict.update(state_updates)
        to.save(state_dict, save_file)

    def load_training_state(self, save_file):
        """Load a previously-saved training state from `save_file`."""
        checkpoint = to.load(save_file)
        # Optimiser state reading depends on model device, so set that here
        self.model.to(device=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # Now PyTorch does a deepcopy of the state_dict, so must manually
        # overwrite hyperparameters to retain identical equality
        self.network_weight_optimiser.load_state_dict(
            checkpoint['network_weight_optimiser_state_dict'])
        for live_param_group, checkpoint_param_group in zip(
                self.network_weight_optimiser.param_groups,
                checkpoint['network_weight_optimiser_state_dict']['param_groups']):
            live_param_group.update({
                key: value
                for key, value in checkpoint_param_group.items()
                if key != 'params'})

        perturbed_initial_config = any(
            next(iter(self.config_dicts['network_weight'][key]))
            != next(iter(checkpoint['initial_network_weight_dict'][key]))
            if isinstance(self.config_dicts['network_weight'][key], repeat)
            else False  # Not implemented for non-scalar hyperparameters
            for key in self.config_dicts['network_weight'])
        if self._ray_tune_config and perturbed_initial_config:
            # Ray Tune has perturbed the initial hyperparameters.
            # Work out by what ratio, then apply the same perturbation to our
            # tuned hyperparameters, and use those
            perturbation_ratios = {
                key:
                next(iter(self.config_dicts['network_weight'][key]))
                / next(iter(checkpoint['initial_network_weight_dict'][key]))
                for key in checkpoint['initial_network_weight_dict']
                if key != 'class'}
            self.config_dicts['network_weight'].update(
                checkpoint['network_weight_dict'])

            for key, ratio in perturbation_ratios.items():
                value_iterable = self.config_dicts['network_weight'][key]
                assert isinstance(value_iterable, repeat)
                value = next(iter(value_iterable))
                if isinstance(value, to.Tensor):
                    value.data *= ratio
                else:
                    self.config_dicts['network_weight'][key] = repeat(value * ratio)
        else:
            self.config_dicts['network_weight'].update(
                checkpoint['network_weight_dict'])

        if 'hyperparameter_optimiser_state_dict' in checkpoint:
            self.hyperparameter_optimiser.load_state_dict(
                checkpoint['hyperparameter_optimiser_state_dict'])

        if self._ray_tune_config:
            # We're restarting from an arbitrary point, rather than just
            # repeating the same number of steps as for Penn Treebank, so load
            # current optimisation step indices too
            self.network_weight_step = checkpoint['network_weight_step']
            self.hyperparameter_step = checkpoint['hyperparameter_step']


def main(config_dict=None, config_override={}):
    """Main entry point for executing a training run.
    """
    if not config_dict:
        config_dict = config.load_config()
    util.nested_update(config_dict, config_override)
    load_state = config_dict.pop('load_state')
    save_state = config_dict.pop('save_state')

    # Must always call config.log_directory() to clean up config
    log_directory = config.log_directory(config_dict)
    if load_state and not config_dict.get('_ray_tune_config', False):
        log_directory = to.load(load_state)['log_directory']
    process_id = config.parallel_process_id()

    with SummaryWriter(log_dir=log_directory) as tracker:
        # Include inside the Tensorboard context so the necessary directories
        # are created
        config_path = os.path.join(log_directory, 'config.json')
        if os.path.exists(config_path):
            config_path = os.path.join(log_directory, 'config_checkpoint.json')
        with open(config_path, 'w') as config_file:
            json.dump(config_dict, config_file)
        config_dict.pop('algorithm', None)

        tracker.add_pytorch_tensor = lambda name, value: to.save(
            value, os.path.join(log_directory, name))
        weights_path = os.path.join(log_directory, 'Weights')
        if not os.path.exists(weights_path):
            os.mkdir(weights_path)

        learner = Learner(tensorboard_tracker=tracker,
                          process_id=process_id,
                          **config_dict)
        if load_state:
            learner.load_training_state(load_state)
        learner.train()
        if save_state:
            learner.save_training_state(save_state)
    # Compute fresh validation and test losses to return
    return (learner._compute_loss(next(iter(learner.full_validation_dataloader)),
                                  'Validation').item(),
            learner._compute_loss(next(iter(learner.test_dataloader)),
                                  'Test').item())


# Python <=3.8 uses a relative __file__; force it to be absolute
__file__ = os.path.abspath(__file__)
if __name__ == '__main__':
    main()

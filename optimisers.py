from itertools import chain

import torch as to
import higher
from torch.optim import SGD, Adam


class ExactHypergradientOptimiser():
    """Hypergradient optimiser using the gradients of the validation loss
    directly.
    """

    def __init__(self,
                 base_optimiser):
        self.base_optimiser = base_optimiser

    def step(self, _, validation_loss):
        validation_loss.backward()
        self.base_optimiser.step()

    def zero_grad(self):
        """Zero the gradients of the child optimiser.
        """
        self.base_optimiser.zero_grad()

    def state_dict(self):
        state_dict = self.base_optimiser.state_dict()
        # Must specifically save params for hyperparameter optimiser to retain
        # the link with identically equal objects elsewhere
        state_dict['params'] = [group['params']
                                for group in self.base_optimiser.param_groups]
        return state_dict

    def load_state_dict(self, state_dict):
        params = state_dict.pop('params')
        result = self.base_optimiser.load_state_dict(state_dict)
        for group, param_list in zip(self.base_optimiser.param_groups, params):
            group['params'] = param_list
        return result


class BaydinHypergradientDescent(ExactHypergradientOptimiser):
    """Implementation of the hyperparameter components of hypergradient descent
    after SGD-HD from Baydin et al (2018).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hyperparameters = list(
            chain(
                *(g['params']
                  for g in self.base_optimiser.param_groups)))

    def step(self, last_loss, new_loss):
        # DIFFERENCE:
        # Baydin et al (2018) use two different data minibatches to compute
        # these gradients: the minibatches used for the previous two network
        # weight updates.
        # We also use two different data minibatches to compute these
        # gradients, but NOT necessarily the same as used for the
        # previous two network weight updates.
        # This aids our implementation, but should not hamper this algorithm's
        # performance.
        new_grads = to.autograd.grad(
            new_loss, self.functional_model.parameters(time=-1))
        last_grads = to.autograd.grad(
            last_loss, self.functional_model.parameters(time=-2))
        # Dot product gradients across all parameters
        hypergrad = sum(to.sum(new * -last)
                        for new, last in zip(new_grads, last_grads))

        for hyperparameter, new_grad, last_grad in zip(
                self.hyperparameters, new_grads, last_grads):
            hyperparameter.grad = hypergrad
        self.base_optimiser.step()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['hyperparameters'] = self.hyperparameters
        return state_dict

    def load_state_dict(self, state_dict):
        self.hyperparameters = state_dict.pop('hyperparameters', self.hyperparameters)
        return super().load_state_dict(state_dict)


class NeumannHypergradientOptimiser(ExactHypergradientOptimiser):
    """Wrapper optimiser for updating hyperparameters based on a general update
    function u.
    """

    max_useful_parameter_age = 2

    def __init__(self,
                 base_optimiser,
                 hyperparameter_rollback_distance):
        super().__init__(base_optimiser)
        self.hyperparameters = list(
            chain(
                *(g['params']
                  for g in self.base_optimiser.param_groups)))

        self.hyperparameter_rollback_distance = hyperparameter_rollback_distance

        self.functional_model = None
        self.first_update = True

    @property
    def network_weights(self):
        return self.functional_model.parameters(time=-1)

    @property
    def last_network_weights(self):
        return self.functional_model.parameters(time=-2)

    def step(self, _, validation_loss):
        """Perform a single hyperparameter optimisation step.
        """
        hypergradients = self._hypergradient(validation_loss)

        # Install hypergradients in hyperparameters
        for hyperparameter, hypergradient in zip(
                self.hyperparameters, hypergradients):
            hyperparameter.grad = hypergradient

        self.base_optimiser.step()

    def _hypergradient(self, validation_loss):
        validation_grad_weights = to.autograd.grad(validation_loss,
                                                   self.last_network_weights,
                                                   retain_graph=True)
        direct_gradient = to.autograd.grad(validation_loss,
                                           self.hyperparameters,
                                           allow_unused=True,
                                           retain_graph=True)
        direct_gradient = [to.zeros_like(h) if g is None
                           else g
                           for g, h in zip(direct_gradient,
                                           self.hyperparameters)]

        # Indirect gradient
        network_weight_update = [new - old
                                 for new, old in
                                 zip(self.network_weights,
                                     self.last_network_weights)]

        # Start inverse Hessian product
        summation_term = validation_grad_weights
        approximate_inverse_hessian_product = summation_term
        for _ in range(self.hyperparameter_rollback_distance):
            summation_term = [
                st - grad
                for st, grad in zip(
                        summation_term,
                        to.autograd.grad(network_weight_update,
                                         self.last_network_weights,
                                         grad_outputs=summation_term,
                                         retain_graph=True))]
            approximate_inverse_hessian_product = [
                aihp + term
                for aihp, term in zip(approximate_inverse_hessian_product,
                                      summation_term)]
        # End inverse Hessian product

        indirect_gradient = to.autograd.grad(
            network_weight_update,
            self.hyperparameters,
            grad_outputs=approximate_inverse_hessian_product,
            allow_unused=self.first_update)
        indirect_gradient = [to.zeros_like(h) if g is None
                             else g
                             for g, h in zip(indirect_gradient,
                                             self.hyperparameters)]

        self.first_update = False
        return [direct + indirect
                for direct, indirect in zip(
                        direct_gradient, indirect_gradient)]

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict['first_update'] = self.first_update
        state_dict['hyperparameters'] = self.hyperparameters
        return state_dict

    def load_state_dict(self, state_dict):
        self.first_update = state_dict.pop('first_update', self.first_update)
        self.hyperparameters = state_dict.pop('hyperparameters', self.hyperparameters)
        return super().load_state_dict(state_dict)


class CurvatureTransformingSGD(to.optim.Optimizer):
    """Reimplementation of PyTorch SGD to support curvature-aware matrix-based
    learning rate transformations.
    """

    def __init__(self, params, curvature_transform=None, lr=1):
        defaults = dict(lr=lr,
                        curvature_transform=curvature_transform)
        super().__init__(params, defaults)


class DifferentiableCurvatureTransformingSGD(higher.optim.DifferentiableOptimizer):
    """Reimplementation of PyTorch SGD to support curvature-aware matrix-based
    learning rate transformations.
    """

    def _update(self, grouped_grads):
        """Performs a single optimisation step.
        """
        zipped = zip(self.param_groups, grouped_grads)
        for group_index, (group, grads) in enumerate(zipped):
            for param_index, (param, grad) in enumerate(zip(group['params'], grads)):
                if grad is None:
                    continue

                curvature_transform = group['curvature_transform'].tril()
                curvature_transform = curvature_transform @ curvature_transform.T
                self.last_curvature_transform = curvature_transform

                param_shape = param.shape
                param_update = group['lr'] * curvature_transform @ grad.view(-1)
                param_update = param_update.reshape(param_shape)
                param = param.add(-param_update)
                group['params'][param_index] = param


higher.register_optim(CurvatureTransformingSGD,
                      DifferentiableCurvatureTransformingSGD)

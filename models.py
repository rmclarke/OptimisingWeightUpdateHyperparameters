"""Definitions of models, loss functions and helpers.
"""
import os

import torch as to
import numpy as np
import scipy.stats
from torch.nn import CrossEntropyLoss, MSELoss
from torchvision.models import resnet18

import util


class MultiLayerPerceptron(to.nn.Module):
    """Implement a multi-layer percepron model, comprising arbitrarily many
    hidden layers of varying size and one activation function throughout.
    """

    def __init__(self,
                 layer_sizes,
                 activation_function='',
                 batch_norm=False,
                 random_seed=None):
        super().__init__()
        activation_function = getattr(to.nn, activation_function, None)

        if random_seed:
            to.manual_seed(random_seed)
        # Initialise layers with the first layer, so we cleanly avoid an
        # activation function on the final layer
        layers = [to.nn.Linear(layer_sizes[0], layer_sizes[1])]
        for input_size, output_size in zip(layer_sizes[1:], layer_sizes[2:]):
            if batch_norm:
                layers.append(to.nn.BatchNorm1d(input_size))
            layers.append(activation_function())
            layers.append(
                to.nn.Linear(input_size, output_size))
        self.layers = to.nn.Sequential(*layers)

    def forward(self, data):
        return self.layers(data)


class RotatedNoisyQuadratic(to.nn.Module):
    """Implementation of a generic quadratic function in arbitrary dimensions.
    """

    def __init__(self,
                 dimensions,
                 curvature_initialisation,
                 initialise_weights_from_curvature_file=False,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.dimensions = dimensions
        weights = to.nn.Parameter(to.rand(dimensions, dtype=to.double) * 10 - 5)
        self.register_parameter('weights', weights)

        if initialise_weights_from_curvature_file:
            assert os.path.exists(curvature_initialisation)

        if curvature_initialisation == 'random':
            self.curvatures = 4.975 * (to.cos(to.rand(dimensions) * np.pi)
                                       + 10.05/9.95)
            self.variances = 1 / self.curvatures.double()
            self.sqrt_variances = to.sqrt(self.variances)
            eigenvalues = np.float64(np.random.rand(dimensions)) * 3
            eigenvalues = dimensions * (eigenvalues / eigenvalues.sum())
            # Tweak for stability
            eigenvalues[-1] = dimensions - eigenvalues[:-1].sum()
            self.correlations = to.from_numpy(
                scipy.stats.random_correlation.rvs(eigenvalues))
            self.full_curvature = to.diag(self.sqrt_variances) @ self.correlations @ to.diag(self.sqrt_variances)
        elif isinstance(curvature_initialisation, list):
            self.full_curvature = to.tensor(curvature_initialisation,
                                            dtype=to.double)
        elif (isinstance(curvature_initialisation, str)
              and os.path.exists(curvature_initialisation)):
            data_cache = util.get_tags(curvature_initialisation)
            self.full_curvature = to.zeros(2, 2, dtype=to.double)
            for i in range(2):
                for j in range(2):
                    self.full_curvature[i, j] = data_cache[f'True_Inverse_Hessian/{i},{j}'][0][0]
                if initialise_weights_from_curvature_file:
                    self.weights.data[i] = data_cache[f'Weights/{i}'][0][0]
        else:
            raise ValueError(f'Unknown curvature_initialisation: {curvature_initialisation}')

        self.hessian = self.full_curvature.inverse()

    def to(self, *args, device=None, **kwargs):
        super().to(*args, device, **kwargs)
        self.full_curvature = self.full_curvature.to(device)
        self.hessian = self.hessian.to(device)

    def forward(self, *_):
        # perturbations = to.randn_like(self.weights) * 0.1
        perturbations = 0
        return 0.5 * ((self.weights - perturbations).T
                      @ self.hessian
                      @ (self.weights - perturbations)) / self.dimensions


class LSTMwithEmbedding(to.nn.LSTM):
    """Wrapper around the built-in LSTM model, adding a pre-encoder and a
    post-decoder.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_tokens,
                 *args,
                 **kwargs):
        super().__init__(input_size, hidden_size, *args, **kwargs)
        self.encoder = to.nn.Embedding(num_tokens, input_size)
        self.decoder = to.nn.Linear(hidden_size, num_tokens)

        self.num_tokens = num_tokens

    def forward(self, input_sequence, *hidden_state):
        encoded_input = self.encoder(input_sequence)

        # Need to disable CuDNN to support second-derivative backward() calls
        with to.backends.cudnn.flags(enabled=False):
            predictions, hidden_state = super().forward(
                encoded_input, hidden_state if hidden_state else None)

        decoded_predictions = self.decoder(predictions)
        return decoded_predictions.permute(0, 2, 1), hidden_state


class IdentityParameterModel(to.nn.Module):
    """A non-op model which directly returns its parameters."""

    def __init__(self, initial_weights):
        super().__init__()
        weights = to.nn.Parameter(
            to.tensor(initial_weights))
        self.register_parameter('weights', weights)

    def forward(self, *_):
        return self.weights


class IdentityLoss(to.nn.Module):

    def forward(self, predictions, _):
        return predictions


class SumLoss(to.nn.Module):

    def forward(self, predictions, _):
        return predictions.sum()


class SqrtSumLoss(to.nn.Module):

    def forward(self, predictions, _):
        return to.sqrt(predictions.sum())


class PartialSqrtSumLoss(to.nn.Module):

    def forward(self, predictions, _):
        predictions_sum = predictions.sum()
        if predictions_sum >= 1:
            return to.sqrt(predictions_sum)
        else:
            return predictions_sum


class AnalyticalToyLoss(to.nn.modules.loss._Loss):
    """Wrapper for the separate training and validation losses of the
    analytical toy problem.
    """

    def forward(self, inputs, targets):
        w1, w2 = inputs
        if targets == 0:
            # Training loss
            return (w1 + to.sin(w2))**2 + 0.2*(w2 + 2)**2
        else:
            # Validation loss
            return (w1 - 1)**2 + (w2 - 1)**2

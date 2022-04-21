"""Definitions and helpers for datasets."""

import numpy as np
import os
import torch as to
import torchvision as tv
import torchvision.transforms as tvt


def make_split_datasets(dataset, validation_proportion=0, **dataset_kwargs):
    normalise_inputs = dataset_kwargs.pop('normalise_inputs', False)
    normalise_outputs = dataset_kwargs.pop('normalise_outputs', False)

    train_val_dataset = dataset(train=True, **dataset_kwargs)
    train_val_sizes = len(train_val_dataset) * np.array(
        [1 - validation_proportion, validation_proportion])
    train_val_sizes = np.rint(train_val_sizes).astype(np.int)
    # If proportional split isn't exact, may need to adjust indices to avoid
    # overflowing the dataset
    train_val_sizes[-1] -= sum(train_val_sizes) - len(train_val_dataset)
    test_dataset = dataset(train=False, **dataset_kwargs)

    if normalise_inputs:
        input_data = to.stack([point[0] for point in train_val_dataset])
        normalise_dimension = (0, 1) if input_data.ndim == 3 else 0
        means = input_data.mean(dim=normalise_dimension)
        standard_deviations = input_data.std(dim=normalise_dimension)
        standard_deviations[standard_deviations == 0] = 1
        normaliser = Normaliser(means, standard_deviations)
        for dataset in train_val_dataset, test_dataset:
            if isinstance(dataset.transform, tvt.Compose):
                dataset.transform.transforms.append(normaliser)
            elif dataset.transform is not None:
                dataset.transform = tvt.Compose([dataset.transform,
                                                 normaliser])
            else:
                dataset.transform = normaliser

    if normalise_outputs:
        output_data = to.stack([point[1] for point in train_val_dataset])
        normalise_dimension = (0, 1)
        means = output_data.mean(dim=normalise_dimension)
        standard_deviations = output_data.std(dim=normalise_dimension)
        standard_deviations[standard_deviations == 0] = 1
        normaliser = Normaliser(means, standard_deviations)
        for dataset in train_val_dataset, test_dataset:
            if isinstance(dataset.target_transform, tvt.Compose):
                dataset.target_transform.transforms.append(normaliser)
            elif dataset.target_transform is not None:
                dataset.target_transform = tvt.Compose([dataset.target_transform,
                                                        normaliser])
            else:
                dataset.target_transform = normaliser
            setattr(dataset, 'target_unnormaliser',
                    lambda x: (x * standard_deviations) + means)

    # TODO: Ensure reproducibility of random sampler
    if validation_proportion == 0:
        return (to.utils.data.Subset(train_val_dataset, range(len(train_val_dataset))),
                NullDataset(),
                test_dataset)
    else:
        return (to.utils.data.Subset(train_val_dataset, range(train_val_sizes[0])),
                to.utils.data.Subset(train_val_dataset, range(train_val_sizes[0],
                                                              sum(train_val_sizes))),
                test_dataset)


def repeating_dataloader(dataloader, reset_callback=lambda: None):
    """Wrapper to create a reloading, infinitely repeating iterator over
    `dataloader`.
    """
    while True:
        for batch in dataloader:
            yield batch
        reset_callback()


class Normaliser():
    """Reimplementation of the torchvision `Normalize` transform, supporting a
    broader range of data sizes.
    """

    def __init__(self, means, standard_deviations):
        self.means = means
        self.standard_deviations = standard_deviations

    def __call__(self, unnormalised_data):
        return (unnormalised_data - self.means) / self.standard_deviations


class NullDataset(to.utils.data.Dataset):

    def __init__(self, *_, train=True, **__):
        super().__init__()
        self.train = train

    def __getitem__(self, index):
        if self.train:
            target = index
        else:
            target = to.tensor(2)
        return to.tensor(0), target

    def __len__(self):
        # Ensure we can still 'split' this dataset for a non-empty validation
        # dataloader
        return 2


class ExternalDataset(to.utils.data.Dataset):
    has_target_data = True

    def __init__(self,
                 root,
                 train=True,
                 transform=None,
                 target_transform=None,
                 conversion=lambda x: x.float()):
        super().__init__()

        self.data = conversion(
            to.load(os.path.join(root, 'data.pt')))
        if self.has_target_data:
            self.targets = conversion(
                to.load(os.path.join(root, 'targets.pt')))
        else:
            self.targets = []

        if train:
            self.data = self.data[type(self).train_val_slice]
            self.targets = self.targets[type(self).train_val_slice]
        else:  # Test
            self.data = self.data[type(self).test_slice]
            self.targets = self.targets[type(self).test_slice]

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        data = self.data[index]
        targets = self.targets[index]

        if self.transform:
            data = self.transform(data)
        if self.target_transform:
            targets = self.target_transform(targets)

        return data, targets


class PennTreebank(ExternalDataset):

    track_perplexities = True
    has_target_data = False
    train_val_slice = slice(1003349)
    test_slice = slice(-82430, None)

    def __init__(self, *args, parallel_sequences, **kwargs):
        super().__init__(*args, root='./data/PennTreebank', **kwargs, conversion=lambda x: x.long())
        self._unbatched_data = self.data.clone()
        self.batchify(parallel_sequences)
        self.targets = self.data[1:]
        self.data = self.data[:-1]

    def batchify(self, parallel_sequences):
        """Rearrange the dataset into batches of `parallel_sequences`."""
        # Permit rebatchifying with a different `parallel_sequences`
        # by unbatchifying first
        self.data = self._unbatched_data.clone()
        excess_tokens = self.data.shape[0] % parallel_sequences
        if excess_tokens != 0:
            self.data = self.data[:-excess_tokens]
        self.data = (self.data
                     .view(parallel_sequences, -1)
                     .t()
                     .contiguous())

    @staticmethod
    def preprocess_data(path='./data/PennTreebank'):
        """Parse the plain-text, raw source into a more computationally useful
        format. Follows example of
        https://github.com/salesforce/awd-lstm-lm/blob/master/data.py
        """
        all_words = set()
        with open(os.path.join(path, 'data.txt'), 'r') as f:
            for line in f:
                words = line.split() + ['<eos>']
                all_words.update(words)
        with open(os.path.join(path, 'data.txt'), 'r') as f:
            word_ids = dict(zip(all_words, range(len(all_words))))
            encoded_words = to.tensor(
                [word_ids[word]
                 for line in f
                 for word in line.split() + ['<eos>']])
        to.save(encoded_words, os.path.join(path, 'data.pt'))
        to.save(word_ids, os.path.join(path, 'word_ids.pt'))


class CIFAR10(ExternalDataset):

    track_accuracies = True
    train_val_slice = slice(50000)
    test_slice = slice(-10000, None)

    def __init__(self, *args, **kwargs):
        # NOTE: These normaliser means and standard deviations are only correct
        # if the pixel values have already been rescaled from [0, 255] to
        # [0, 1], which our code does not do. Since batch normalisation in
        # ResNet-18 will largely mitigate this error, so our published results
        # remain valid, we leave this code as-is to allow for reproducibility.
        # For any other use case, either divide the input data by 255 or update
        # these normalisation coefficients.
        super().__init__(*args,
                         root='./data/CIFAR10',
                         transform=tvt.Compose([
                             tvt.Normalize((0.4914, 0.4822, 0.4465),
                                           (0.247, 0.243, 0.261))]),
                         **kwargs)
        self.targets = self.targets.to(dtype=to.long)


class FashionMNIST(ExternalDataset):

    track_accuracies = True
    train_val_slice = slice(60000)
    test_slice = slice(-10000, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/FashionMNIST', **kwargs)
        self.targets = self.targets.to(dtype=to.long)


class UCI_Boston(ExternalDataset):

    train_val_slice = slice(364 + 91)
    test_slice = slice(-51, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Boston', **kwargs)


class UCI_Concrete(ExternalDataset):

    train_val_slice = slice(742 + 185)
    test_slice = slice(-103, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Concrete', **kwargs)


class UCI_Energy(ExternalDataset):

    train_val_slice = slice(614 + 78)
    test_slice = slice(-76, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Energy', **kwargs)


class UCI_Kin8nm(ExternalDataset):

    train_val_slice = slice(5898 + 1475)
    test_slice = slice(-819, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Kin8nm', **kwargs)


class UCI_Naval(ExternalDataset):

    train_val_slice = slice(8593 + 2148)
    test_slice = slice(-1193, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Naval', **kwargs)


class UCI_Power(ExternalDataset):

    train_val_slice = slice(6889 + 1722)
    test_slice = slice(-957, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Power', **kwargs)


class UCI_Wine(ExternalDataset):

    train_val_slice = slice(1151 + 288)
    test_slice = slice(-160, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Wine', **kwargs)


class UCI_Yacht(ExternalDataset):

    train_val_slice = slice(222 + 55)
    test_slice = slice(-31, None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, root='./data/UCI_Yacht', **kwargs)

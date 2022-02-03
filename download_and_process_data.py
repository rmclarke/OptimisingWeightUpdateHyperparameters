"""Script to construct all the dataset objects required by our code."""

import gzip
import os
import pickle
import shutil
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import torch as to


DATA_ROOT = './data'
UCI_DOWNLOAD_MAP = {'Boston': 'bostonHousing',
                    'Concrete': 'concrete',
                    'Energy': 'energy',
                    'Kin8nm': 'kin8nm',
                    'Naval': 'naval-propulsion-plant',
                    'Power': 'power-plant',
                    'Wine': 'wine-quality-red',
                    'Yacht': 'yacht'}
UCI_DOWNLOAD_URL = 'https://raw.githubusercontent.com/yaringal/DropoutUncertaintyExps/master/UCI_Datasets/{name}/data/data.txt'


def download_file(url, path):
    """Download the file at `url` and save it to `path`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with (urllib.request.urlopen(url) as response,
          open(path, 'wb') as download_file):
        shutil.copyfileobj(response, download_file)


def get_uci_dataset(name):
    """Download and parse the dataset UCI_`name`."""
    dataset_root = os.path.join(DATA_ROOT, f'UCI_{name}')
    download_file(UCI_DOWNLOAD_URL.format(name=UCI_DOWNLOAD_MAP[name]),
                  os.path.join(dataset_root, 'data_targets.txt'))

    raw_data = np.loadtxt(
        os.path.join(dataset_root, 'data_targets.txt'))
    permutation_indices = np.loadtxt(
        os.path.join(dataset_root, 'permutation_indices.txt'),
        dtype='int')
    if name == 'Naval':
        # Following https://github.com/yaringal/DropoutUncertaintyExps, we use
        # the first 16 columns as input data, the 17th as target data, and
        # ignore the 18th
        raw_data = raw_data[:, :-1]

    permuted_data = to.from_numpy(raw_data[permutation_indices])
    to.save(permuted_data[:, :-1], os.path.join(dataset_root, 'data.pt'))
    to.save(permuted_data[:, -1:], os.path.join(dataset_root, 'targets.pt'))


def get_fashion_mnist():
    """Download and parse the Fashion-MNIST dataset."""
    dataset_root = os.path.join(DATA_ROOT, 'FashionMNIST')
    download_file('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz',
                  os.path.join(dataset_root, 'train_data.gz'))
    download_file('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz',
                  os.path.join(dataset_root, 'train_targets.gz'))
    download_file('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz',
                  os.path.join(dataset_root, 'test_data.gz'))
    download_file('http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz',
                  os.path.join(dataset_root, 'test_targets.gz'))

    # Logic borrowed from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    with gzip.open(os.path.join(dataset_root,
                                'train_data.gz'), 'rb') as raw_path:
        train_data = np.frombuffer(raw_path.read(),
                                   dtype=np.uint8,
                                   offset=16).reshape(-1, 784)
    with gzip.open(os.path.join(dataset_root,
                                'train_targets.gz'), 'rb') as raw_path:
        train_targets = np.frombuffer(raw_path.read(),
                                      dtype=np.uint8,
                                      offset=8)
    with gzip.open(os.path.join(dataset_root,
                                'test_data.gz'), 'rb') as raw_path:
        test_data = np.frombuffer(raw_path.read(),
                                  dtype=np.uint8,
                                  offset=16).reshape(-1, 784)
    with gzip.open(os.path.join(dataset_root,
                                'test_targets.gz'), 'rb') as raw_path:
        test_targets = np.frombuffer(raw_path.read(),
                                     dtype=np.uint8,
                                     offset=8)

    data = to.cat((to.from_numpy(train_data),
                   to.from_numpy(test_data)),
                  dim=0)
    targets = to.cat((to.from_numpy(train_targets),
                      to.from_numpy(test_targets)),
                     dim=0)
    to.save(data, os.path.join(dataset_root, 'data.pt'))
    to.save(targets, os.path.join(dataset_root, 'targets.pt'))


def get_cifar10():
    """Download and parse the CIFAR-10 dataset."""
    dataset_root = os.path.join(DATA_ROOT, 'CIFAR10')
    download_file('https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                  os.path.join(dataset_root, 'base_archive.tar.gz'))

    # Logic informed by https://cs.toronto.edu/~kriz/cifar.html
    with tarfile.open(
            os.path.join(dataset_root,
                         'base_archive.tar.gz'), 'r') as archive_file:
        archive_file.extractall(dataset_root)

    data, targets = [], []
    batch_files = ('data_batch_1',
                   'data_batch_2',
                   'data_batch_3',
                   'data_batch_4',
                   'data_batch_5',
                   'test_batch')
    for batch_file in batch_files:
        batch_path = os.path.join(dataset_root,
                                  'cifar-10-batches-py',
                                  batch_file)
        with open(batch_path, 'rb') as batch_data:
            batch_dict = pickle.load(batch_data, encoding='bytes')
        data.append(
            to.from_numpy(
                batch_dict[b'data']
                .reshape(-1, 3, 32, 32)))
        targets.append(
            to.tensor(
                batch_dict[b'labels']))

    data = to.cat(data, dim=0)
    targets = to.cat(targets, dim=0)
    to.save(data, os.path.join(dataset_root, 'data.pt'))
    to.save(targets, os.path.join(dataset_root, 'targets.pt'))


def get_penntreebank():
    """Download and parse the PennTreebank dataset."""
    dataset_root = os.path.join(DATA_ROOT, 'PennTreebank')
    download_file('http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz',
                  os.path.join(dataset_root, 'base_archive.tar.gz'))

    with tarfile.open(
            os.path.join(dataset_root,
                         'base_archive.tar.gz'), 'r') as archive_file:
        archive_file.extractall(dataset_root)
    with open(os.path.join(dataset_root,
                           'data.txt'), 'wb') as output_file:
        for src_name in ('train', 'valid', 'test'):
            with open(os.path.join(dataset_root,
                                   'simple-examples',
                                   'data',
                                   f'ptb.{src_name}.txt'), 'rb') as input_file:
                shutil.copyfileobj(input_file, output_file)

    word_ids = to.load(os.path.join(dataset_root, 'word_ids.pt'))
    with open(os.path.join(dataset_root, 'data.txt'), 'r') as data_file:
        encoded_words = to.tensor(
            [word_ids[word]
             for line in data_file
             for word in line.split() + ['<eos>']])
    to.save(encoded_words, os.path.join(dataset_root, 'data.pt'))


def main():
    """Main function downloading and processing all datasets."""
    for uci_name in UCI_DOWNLOAD_MAP:
        get_uci_dataset(uci_name)
    get_fashion_mnist()
    get_cifar10()
    get_penntreebank()


if __name__ == '__main__':
    main()

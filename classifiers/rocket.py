# Angus Dempster, Francois Petitjean, Geoff Webb

# Dempster A, Petitjean F, Webb GI (2019) ROCKET: Exceptionally fast and
# accurate time series classification using random convolutional kernels.
# arXiv:1910.13051
import time

import numpy as np
from numba import njit, prange
from sklearn.linear_model import RidgeClassifierCV

from classifiers.classifiers import predict_model
from utils.tools import calculate_metrics


@njit
def generate_kernels(input_length, num_kernels, num_channels=1):
    candidate_lengths = np.array((7, 9, 11), dtype=np.int32)
    candidate_lengths = candidate_lengths[candidate_lengths < input_length]
    lengths = np.random.choice(candidate_lengths, num_kernels)

    num_channel_indices = (2 ** np.random.uniform(0, np.log2(num_channels + 1), num_kernels)).astype(
        np.int32)  # exponential
    # num_channel_indices = np.random.randint(1, num_channels + 1, num_kernels) # uniform
    # num_channel_indices[num_channel_indices > 3] = 3 # limit
    channel_indices = np.zeros(num_channel_indices.sum(), dtype=np.int32)

    weights = np.zeros((num_channels, lengths.sum()), dtype=np.float32)
    biases = np.zeros(num_kernels, dtype=np.float32)
    dilations = np.zeros(num_kernels, dtype=np.int32)
    paddings = np.zeros(num_kernels, dtype=np.int32)

    for i in range(num_kernels):

        _weights = np.random.normal(0, 1, (num_channels, lengths[i]))

        a = lengths[:i].sum()
        b = a + lengths[i]
        for j in range(num_channels):
            _weights[j] = _weights[j] - _weights[j].mean()
        weights[:, a:b] = _weights

        a1 = num_channel_indices[:i].sum()
        b1 = a1 + num_channel_indices[i]
        channel_indices[a1:b1] = np.random.choice(np.arange(0, num_channels), num_channel_indices[i], replace=False)

        biases[i] = np.random.uniform(-1, 1)

        dilation = 2 ** np.random.uniform(0, np.log2((input_length - 1) // (lengths[i] - 1)))
        dilation = np.int32(dilation)
        dilations[i] = dilation

        padding = ((lengths[i] - 1) * dilation) // 2 if np.random.randint(2) == 1 else 0
        paddings[i] = padding

    return weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices


@njit(fastmath=True)
def apply_kernel(X, weights, length, bias, dilation, padding, num_channel_indices, channel_indices, stride):
    # zero padding
    if padding > 0:
        _input_length, _num_channels = X.shape
        _X = np.zeros((_input_length + (2 * padding), _num_channels))
        _X[padding:(padding + _input_length), :] = X
        X = _X

        # input_length = len(X)
    input_length, num_channels = X.shape

    output_length = input_length - ((length - 1) * dilation)

    _ppv = 0
    _max = np.NINF

    for i in range(0, output_length, stride):
        _sum = bias

        for j in range(length):
            for k in range(num_channel_indices):
                _sum += weights[channel_indices[k], j] * X[i + (j * dilation), channel_indices[k]]

        if _sum > _max:
            _max = _sum

        if _sum > 0:
            _ppv += 1

    return _ppv / output_length, _max


@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels, stride=1):
    weights, lengths, biases, dilations, paddings, num_channel_indices, channel_indices = kernels

    num_examples = len(X)
    num_kernels = len(lengths)

    _X = np.zeros((num_examples, num_kernels * 2), dtype=np.float32)  # 2 features per kernel

    for i in prange(num_examples):
        a = 0
        a1 = 0
        for j in range(num_kernels):
            b = a + lengths[j]
            b1 = a1 + num_channel_indices[j]

            _X[i, (j * 2):((j * 2) + 2)] = \
                apply_kernel(X[i], weights[:, a:b], lengths[j], biases[j], dilations[j], paddings[j],
                             num_channel_indices[j], channel_indices[a1:b1], stride)

            a = b
            a1 = b1

    return _X


class Classifier_Rocket:
    def __init__(self, output_directory, input_shape, nb_classes, verbose):
        if verbose:
            print("[Rocket] Creating Rocket classifier")

        self.verbose = verbose
        self.output_directory = output_directory
        self.input_shape = input_shape
        self.nb_classes = nb_classes

    def fit(self, Ximg_train, yimg_train):
        start_time = time.time()
        if self.verbose:
            print('[Rocket] Generating kernels')
        self.kernels = generate_kernels(Ximg_train.shape[1], 10000, Ximg_train.shape[2])

        if self.verbose:
            print('[Rocket] Applying kernels')
        X_training_transform = apply_kernels(Ximg_train, self.kernels)

        if self.verbose:
            print('[Rocket] Training')
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), normalize=True)
        self.classifier.fit(X_training_transform, yimg_train)

        self.duration = time.time() - start_time
        if self.verbose:
            print('[Rocket] Training done!, took {}s'.format(self.duration))

    def predict(self, Ximg, yimg):
        if self.verbose:
            print('[Rocket] Predicting')
        X_test_transform = apply_kernels(Ximg, self.kernels)

        model_metrics, conf_mat, y_true, y_pred = predict_model(self.classifier, X_test_transform, yimg,
                                                                self.output_directory)

        df_metrics = calculate_metrics(y_true, y_pred, self.duration)
        df_metrics.to_csv(self.output_directory + 'df_metrics.csv', index=False)

        if self.verbose:
            print('[Rocket] Prediction done!')
        return model_metrics, conf_mat

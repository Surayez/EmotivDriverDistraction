import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Embedding, Dropout
import numpy as np
import keras


# Reference: https://rubikscode.net/2019/08/19/transformer-with-python-and-tensorflow-2-0-encoder-decoder/

class PositionalEncoding(object):
    def __init__(self, position, d):
        angle_rads = self._get_angles(np.arange(position)[:, np.newaxis], np.arange(d)[np.newaxis, :], d)

        sines = np.sin(angle_rads[:, 0::2])
        cosines = np.cos(angle_rads[:, 1::2])
        self._encoding = np.concatenate([sines, cosines], axis=-1)
        self._encoding = self._encoding[np.newaxis, ...]

    def _get_angles(self, position, i, d):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d))
        return position * angle_rates

    def get_positional_encoding(self):
        return tf.cast(self._encoding, dtype=tf.float32)


class PreProcessingLayer(keras.layers.Layer):
    def __init__(self):
        super(PreProcessingLayer, self).__init__()

        # Initialize
        self.num_neurons = 128

        # Add positional encoding
        positional_encoding_handler = PositionalEncoding(100, self.num_neurons)
        self.positional_encoding = positional_encoding_handler.get_positional_encoding()

        # Add positional encoding
        self.dropout = Dropout(0.1)

    def call(self, sequence):
        sequence_lenght = tf.shape(sequence)[1]

        sequence *= tf.math.sqrt(tf.cast(self.num_neurons, tf.float32))
        sequence += self.positional_encoding[:, :sequence_lenght, :]

        return sequence

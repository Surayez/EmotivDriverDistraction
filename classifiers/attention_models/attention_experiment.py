import keras
import tensorflow as tf
from tensorflow.keras import layers

# from tensor2tensor.models.transformer import Transformer
from tensor2tensor.models import transformer

__author__ = "Surayez Rahman"


# # Ref: https://github.com/tensorflow/tensor2tensor/issues/813
# def transformer_code(inputLayer):
#     hparams = transformer.transformer_base()
#     encoder = transformer.TransformerEncoder(hparams, mode=tf.estimator.ModeKeys.TRAIN)
#     x = keras.backend.expand_dims(inputLayer, axis=2)
#     y = encoder({"inputs": x, "targets": 0, "target_space_id": 0})
#     y = keras.backend.squeeze(y[0], 2)
#     return y
#     # inputAfterDense = keras.layers.Dense(512, activation='relu')(inputLayer)
#     # y = keras.layers.Lambda(transformer_code)(inputAfterDense)


# Ref: https://stackoverflow.com/questions/59718635/how-to-use-tf-lambda-and-tf-variable-at-tensorflow-2-0
class TransformerLayer(layers.Layer):
    def __init__(self):
        super(TransformerLayer, self).__init__()

    def call(self, inputs, **kwargs):
        hparams = transformer.transformer_base()
        encoder = transformer.TransformerEncoder(hparams, mode=tf.estimator.ModeKeys.TRAIN)
        x = keras.backend.expand_dims(inputs, axis=2)
        y = encoder({"inputs": x, "targets": 0, "target_space_id": 0})
        y = keras.backend.squeeze(y[0], 2)
        return y


def build_model(input_shape):
    main_input = keras.layers.Input(input_shape)
    input_layer = keras.layers.Input((input_shape[1], input_shape[2]))

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)
    print(gap_layer)

    cnn_model = keras.layers.TimeDistributed(keras.models.Model(inputs=input_layer, outputs=gap_layer))(main_input)

    # lstm_layer = keras.layers.LSTM(64, return_sequences=True)(cnn_model)
    # lstm_layer = keras.layers.LSTM(64)(lstm_layer)

    # Transformation Layer
    inputAfterDense = keras.layers.Dense(512, activation='relu')(cnn_model)
    print(inputAfterDense)
    transformer_layer = TransformerLayer()(inputAfterDense)
    print(transformer_layer)

    gap_layer1 = keras.layers.pooling.GlobalAveragePooling1D()(transformer_layer)
    print(gap_layer1)

    output_layer = keras.layers.Dense(64, activation='relu')(gap_layer1)
    output_layer = keras.layers.Dense(2, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=main_input, outputs=output_layer)
    model.summary()
    return model


if __name__ == "__main__":
    input_shape = (None, 40, 266)
    attn_model = build_model(input_shape)

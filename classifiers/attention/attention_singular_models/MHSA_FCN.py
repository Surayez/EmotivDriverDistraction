import keras
from keras_multi_head import MultiHeadAttention

__author__ = "Surayez Rahman"


def build_model(input_shape):
    input_layer = keras.layers.Input(input_shape)

    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.normalization.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)

    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.normalization.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)

    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.normalization.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)

    # Attention

    lstm_layer = keras.layers.LSTM(128, return_sequences=True)(conv3)
    att_layer = MultiHeadAttention(head_num=128)(lstm_layer)
    gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(att_layer)

    output_layer = keras.layers.Dense(2, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model


if __name__ == "__main__":
    inp_shape = (40, 266)
    model = build_model(inp_shape)

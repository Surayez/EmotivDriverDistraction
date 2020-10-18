import keras
# from .attention_implements.SeqSelfAttention import SeqSelfAttention
from keras_self_attention import SeqSelfAttention


__author__ = "Surayez Rahman"


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

    cnn_model = keras.layers.TimeDistributed(keras.models.Model(inputs=input_layer, outputs=gap_layer))(main_input)

    lstm_layer = keras.layers.LSTM(64, return_sequences=True)(cnn_model)

    self_attention = SeqSelfAttention(attention_width=input_shape[1], attention_activation='sigmoid', name='Attention',
                                      attention_type=SeqSelfAttention.ATTENTION_TYPE_MUL)(lstm_layer)
    gap_layerX = keras.layers.pooling.GlobalAveragePooling1D()(self_attention)

    output_layer = keras.layers.Dense(64, activation='relu')(gap_layerX)
    output_layer = keras.layers.Dense(2, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=main_input, outputs=output_layer)

    model.summary()
    return model


if __name__== "__main__":
    inp_shape = (None, 40, 266)
    model = build_model(inp_shape)

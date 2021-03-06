import keras
# from .attention_implements.SeqSelfAttention import SeqSelfAttention
from keras_self_attention import SeqSelfAttention

__author__ = "Surayez Rahman"


def build_model(input_shape):
    emotive_params = 266
    n_feature_maps = 64

    main_input = keras.layers.Input(input_shape)
    input_layer = keras.layers.Input((input_shape[1], input_shape[2]))

    lstm_layer = keras.layers.LSTM(emotive_params, return_sequences=True)(input_layer)
    print(lstm_layer)

    self_attention = SeqSelfAttention(attention_width=40, attention_activation='sigmoid', name='Attention')(lstm_layer)
    print(self_attention)

    lstm_layer = keras.layers.LSTM(emotive_params, return_sequences=True)(self_attention)
    print(lstm_layer)

    gap_layerX = keras.layers.pooling.GlobalAveragePooling1D()(lstm_layer)

    # See architecture
    cnn_model = keras.layers.TimeDistributed(keras.models.Model(inputs=input_layer, outputs=gap_layerX))(main_input)
    lstm_layer = keras.layers.LSTM(n_feature_maps, return_sequences=True)(cnn_model)

    gap_layerX = keras.layers.pooling.GlobalAveragePooling1D()(lstm_layer)

    output_layer = keras.layers.Dense(n_feature_maps, activation='relu')(gap_layerX)
    output_layer = keras.layers.Dense(2, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=main_input, outputs=output_layer)

    model.summary()
    return model


if __name__ == "__main__":
    inp_shape = (None, 40, 266)
    model = build_model(inp_shape)


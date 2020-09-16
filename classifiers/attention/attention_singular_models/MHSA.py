import keras
from keras_multi_head import MultiHeadAttention

__author__ = "Surayez Rahman"


def build_model(input_shape):
    emotive_params = 266
    n_feature_maps = 64

    input_layer = keras.layers.Input(input_shape)
    lstm_layer = keras.layers.LSTM(emotive_params, return_sequences=True)(input_layer)

    att_layer = MultiHeadAttention(
        head_num=266,
        name='Multi-Head',
    )(lstm_layer)

    lstm_layer = keras.layers.LSTM(n_feature_maps, return_sequences=True)(att_layer)

    gap_layerX = keras.layers.pooling.GlobalAveragePooling1D()(lstm_layer)

    output_layer = keras.layers.Dense(n_feature_maps, activation='relu')(gap_layerX)
    output_layer = keras.layers.Dense(2, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.summary()
    return model


if __name__ == "__main__":
    inp_shape = (None, 40, 266)
    model = build_model(inp_shape)

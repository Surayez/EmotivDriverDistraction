import keras
from keras_multi_head import MultiHeadAttention


def build_model(input_shape):

    input_layer = keras.layers.Input((input_shape[1], input_shape[2]))
    att_layer = MultiHeadAttention(head_num=2)(input_layer)
    gap_layerX = keras.layers.pooling.GlobalAveragePooling1D()(att_layer)

    output_layer = keras.layers.Dense(64, activation='relu')(gap_layerX)
    output_layer = keras.layers.Dense(2, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)

    model.summary()
    return model


if __name__== "__main__":
    inp_shape = (None, 40, 266)
    model = build_model(inp_shape)

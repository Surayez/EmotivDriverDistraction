import keras
# from .attention_implements.MultiHeadAttention import MultiHeadAttention
from keras_multi_head import MultiHeadAttention

__author__ = "Surayez Rahman"


def build_model(input_shape):
    n_feature_maps = 64

    main_input = keras.layers.Input(input_shape)
    input_layer = keras.layers.Input((input_shape[1], input_shape[2]))

    # add model layerss
    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps,
                                 kernel_size=8,
                                 padding='same')(input_layer)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps,
                                 kernel_size=5,
                                 padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps,
                                 kernel_size=3,
                                 padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps,
                                     kernel_size=1,
                                     padding='same')(input_layer)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2
    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                 kernel_size=8,
                                 padding='same')(output_block_1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                 kernel_size=5,
                                 padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                 kernel_size=3,
                                 padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                     kernel_size=1,
                                     padding='same')(output_block_1)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                 kernel_size=8,
                                 padding='same')(output_block_2)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                 kernel_size=5,
                                 padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                 kernel_size=3,
                                 padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # no need to expand channels because they are equal
    shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    print(output_block_3)

    # FINAL
    gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
    print(gap_layer)

    cnn_model = keras.layers.TimeDistributed(keras.models.Model(inputs=input_layer, outputs=gap_layer))(main_input)
    print(cnn_model)

    lstm_layer = keras.layers.Bidirectional(keras.layers.LSTM(n_feature_maps, return_sequences=True))(cnn_model)
    # lstm_layer = keras.layers.LSTM(n_feature_maps, return_sequences=True)(cnn_model)

    att_layer = MultiHeadAttention(head_num=128)(lstm_layer)

    lstm_layer = keras.layers.LSTM(128, return_sequences=True)(att_layer)

    gap_layerX = keras.layers.pooling.GlobalAveragePooling1D()(lstm_layer)

    output_layer = keras.layers.Dense(64, activation='relu')(gap_layerX)
    output_layer = keras.layers.Dense(2, activation='softmax')(output_layer)

    model = keras.models.Model(inputs=main_input, outputs=output_layer)

    model.summary()
    return model


if __name__ == "__main__":
    inp_shape = (None, 40, 266)
    model = build_model(inp_shape)

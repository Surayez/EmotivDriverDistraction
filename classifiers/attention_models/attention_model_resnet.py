import keras

__author__ = "Surayez Rahman"

# ResNet code here are taken from https://github.com/hfawaz/dl-4-tsc. Attention code taken from:
# https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb


def build_model(input_shape):
    n_feature_maps = 64
    n_hidden = 64

    main_input = keras.layers.Input(input_shape)
    input_layer = keras.layers.Input((input_shape[1], input_shape[2]))

    # BLOCK 1
    conv_x = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_1 = keras.layers.add([shortcut_y, conv_z])
    output_block_1 = keras.layers.Activation('relu')(output_block_1)

    # BLOCK 2

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

    # expand channels for the sum
    shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
    shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

    output_block_2 = keras.layers.add([shortcut_y, conv_z])
    output_block_2 = keras.layers.Activation('relu')(output_block_2)

    # BLOCK 3

    conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
    conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
    conv_x = keras.layers.Activation('relu')(conv_x)

    conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
    conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
    conv_y = keras.layers.Activation('relu')(conv_y)

    conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)

    conv_z = keras.layers.normalization.BatchNormalization()(conv_z)
    shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

    output_block_3 = keras.layers.add([shortcut_y, conv_z])
    output_block_3 = keras.layers.Activation('relu')(output_block_3)
    print(output_block_3)

    # FINAL

    gap_layerX = keras.layers.GlobalAveragePooling1D()(output_block_3)
    print(gap_layerX)

    cnn_model = keras.layers.TimeDistributed(keras.models.Model(inputs=input_layer, outputs=gap_layerX))(main_input)
    print(cnn_model)

    # ATTENTION BLOCK STARTS

    output_shape = 2
    output_train = keras.layers.Input(output_shape)

    # ENCODER [Encodes input to hidden states]
    encoder_stack_h, encoder_last_h, encoder_last_c = keras.layers.LSTM(n_hidden, activation='relu',
                                                                        return_state=True,
                                                                        return_sequences=True)(cnn_model)

    # Returns hidden state stacks and last hidden states
    print(encoder_stack_h)
    print(encoder_last_h)
    print(encoder_last_c)

    # Last hidden states are normalised to prevent gradient explosion
    encoder_last_h = keras.layers.BatchNormalization(momentum=0.6)(encoder_last_h)
    encoder_last_c = keras.layers.BatchNormalization(momentum=0.6)(encoder_last_c)
    print(encoder_last_h)
    print(encoder_last_c)

    # The last hidden state is repeated the number of time the output needs to be predicted
    decoder_input = keras.layers.RepeatVector(output_train.shape[1])(encoder_last_h)
    print(decoder_input)

    # DECODER [Decodes the last encoded hidden state to get alignment scoring]
    decoder_stack_h = keras.layers.LSTM(n_hidden, activation='relu',
                                        return_state=False, return_sequences=True)(decoder_input,
                                                                                   initial_state=[encoder_last_h,
                                                                                                  encoder_last_c])
    print(decoder_stack_h)

    # ATTENTION LAYER
    # Calculates alignment scores, then softmax func
    attention = keras.layers.dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
    attention = keras.layers.Activation('softmax')(attention)
    print(attention)

    # Use the output of Attention to build the context layer
    context = keras.layers.dot([attention, encoder_stack_h], axes=[2, 1])
    context = keras.layers.BatchNormalization(momentum=0.6)(context)
    print(context)

    # Concatenating the context vector and stacked hidden states of decoder
    decoder_combined_context = keras.layers.concatenate([context, decoder_stack_h])
    print(decoder_combined_context)

    gap_layer = keras.layers.GlobalAveragePooling1D()(context)
    print(gap_layer)

    out = keras.layers.Dense(output_train.shape[1])(gap_layer)
    print(out)

    model = keras.models.Model(inputs=main_input, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    model.summary()

    return model


if __name__ == "__main__":
    input_shape = (None, 40, 266)
    attn_model = build_model(input_shape)

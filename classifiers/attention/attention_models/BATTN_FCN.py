import keras

__author__ = "Surayez Rahman"


def build_model(input_shape):
    n_hidden = 64

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

    gap_layerX = keras.layers.GlobalAveragePooling1D()(conv3)

    cnn_model = keras.layers.TimeDistributed(keras.models.Model(inputs=input_layer, outputs=gap_layerX))(main_input)
    print(cnn_model)

    # ATTENTION BLOCK STARTS

    output_shape = 2
    output_train = keras.layers.Input(output_shape)

    # ENCODER [Encodes input to hidden states]
    encoder_stack_h, encoder_last_h, encoder_last_c, x, y = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden, activation='sigmoid',
                                                                        return_state=True,
                                                                        return_sequences=True))(cnn_model)

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
    decoder_stack_h = keras.layers.Bidirectional(keras.layers.LSTM(n_hidden, activation='sigmoid',
                                        return_state=False, return_sequences=True))(decoder_input)
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
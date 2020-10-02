import keras

__author__ = "Surayez Rahman"

# Reference:
# https://levelup.gitconnected.com/building-seq2seq-lstm-with-luong-attention-in-keras-for-time-series-forecasting-1ee00958decb


def build_model(input_shape):
    n_hidden = 64
    output_shape = 2

    input_train = keras.layers.Input(input_shape)
    output_train = keras.layers.Input(output_shape)

    # ENCODER [Encodes input to hidden states]
    encoder_stack_h, encoder_last_h, encoder_last_c = keras.layers.LSTM(n_hidden, activation='elu', dropout=0.2,
                                                                        return_state=True,
                                                                        return_sequences=True)(input_train)
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
    decoder_input = keras.layers.RepeatVector(output_train.shape[1]*2)(encoder_last_h)
    print(decoder_input)

    # DECODER [Decodes the last encoded hidden state to get alignment scoring]
    decoder_stack_h = keras.layers.LSTM(n_hidden, activation='elu', dropout=0.2,
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

    model = keras.models.Model(inputs=input_train, outputs=out)
    model.summary()

    return model


if __name__ == "__main__":
    input_shape = (40, 266)
    attn_model = build_model(input_shape)

import keras


def get_next():
    n_hidden = 100
    input_shape = (40, 266)

    input_train = keras.layers.Input(input_shape)
    output_train = keras.layers.Input(input_shape)

    encoder_stack_h, encoder_last_h, encoder_last_c = keras.layers.LSTM(n_hidden, activation='elu', dropout=0.2,
                                                                        recurrent_dropout=0.2,
                                                                        return_state=True,
                                                                        return_sequences=True)(input_train)
    print(encoder_stack_h)
    print(encoder_last_h)
    print(encoder_last_c)

    encoder_last_h = keras.layers.BatchNormalization(momentum=0.6)(encoder_last_h)
    encoder_last_c = keras.layers.BatchNormalization(momentum=0.6)(encoder_last_c)
    print(encoder_last_h)
    print(encoder_last_c)

    decoder_input = keras.layers.RepeatVector(output_train.shape[1])(encoder_last_h)
    print(decoder_input)

    decoder_stack_h = keras.layers.LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
                                        return_state=False, return_sequences=True)(decoder_input,
                                                                                   initial_state=[encoder_last_h,
                                                                                                  encoder_last_c])
    print(decoder_stack_h)

    attention = keras.layers.dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
    attention = keras.layers.Activation('softmax')(attention)
    print(attention)

    context = keras.layers.dot([attention, encoder_stack_h], axes=[2, 1])
    context = keras.layers.BatchNormalization(momentum=0.6)(context)
    print(context)

    decoder_combined_context = keras.layers.concatenate([context, decoder_stack_h])
    print(decoder_combined_context)

    out = keras.layers.TimeDistributed(keras.layers.Dense(output_train.shape[2]))(decoder_combined_context)
    print(out)

    model = keras.models.Model(inputs=input_train, outputs=out)
    opt = keras.optimizers.Adam(lr=0.01, clipnorm=1)
    model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
    model.summary()


get_next()

# n_hidden = 100
# input_shape = (40, 266)
# output_shape = (40, 2)
#
# input_train = keras.layers.Input(input_shape)
# output_train = keras.layers.Input(input_shape)
#
# encoder_stack_h, encoder_last_h, encoder_last_c = keras.layers.LSTM(n_hidden, activation='elu', dropout=0.2,
#                                                                     recurrent_dropout=0.2,
#                                                                     return_state=True,
#                                                                     return_sequences=True)(input_train)
# print(encoder_stack_h)
# print(encoder_last_h)
# print(encoder_last_c)
#
#
# encoder_last_h = keras.layers.BatchNormalization(momentum=0.6)(encoder_last_h)
# encoder_last_c = keras.layers.BatchNormalization(momentum=0.6)(encoder_last_c)
# print(encoder_last_h)
# print(encoder_last_c)
#
#
# decoder_input = keras.layers.RepeatVector(output_train.shape[1])(encoder_last_h)
# print(decoder_input)
#
#
# decoder_stack_h = keras.layers.LSTM(n_hidden, activation='elu', dropout=0.2, recurrent_dropout=0.2,
#                                     return_state=False, return_sequences=True)(decoder_input, initial_state=[encoder_last_h, encoder_last_c])
# print(decoder_stack_h)
#
#
# attention = keras.layers.dot([decoder_stack_h, encoder_stack_h], axes=[2, 2])
# attention = keras.layers.Activation('softmax')(attention)
# print(attention)
#
#
# context = keras.layers.dot([attention, encoder_stack_h], axes=[2, 1])
# context = keras.layers.BatchNormalization(momentum=0.6)(context)
# print(context)
#
#
# decoder_combined_context = keras.layers.concatenate([context, decoder_stack_h])
# print(decoder_combined_context)
#
#
# out = keras.layers.TimeDistributed(keras.layers.Dense(output_train.shape[2]))(decoder_combined_context)
# print(out)
#
#
# model = keras.models.Model(inputs=input_train, outputs=out)
# opt = keras.optimizers.Adam(lr=0.01, clipnorm=1)
# model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
# model.summary()

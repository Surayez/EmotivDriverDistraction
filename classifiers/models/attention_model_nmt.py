import keras

__author__ = "Surayez Rahman"

# NMT Attention code taken from: https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39

hidden_size = 100
input_shape = (40, 266)
output_shape = 2

# Define an input sequence and process it.
encoder_inputs = keras.layers.Input(shape=input_shape, name='encoder_inputs')
decoder_inputs = keras.layers.Input(shape=output_shape, name='decoder_inputs')

# Encoder GRU
encoder_gru = keras.layers.GRU(hidden_size, return_sequences=True, return_state=True, name='encoder_gru')
encoder_out, encoder_state = encoder_gru(encoder_inputs)

# Set up the decoder GRU, using `encoder_states` as initial state.
decoder_gru = keras.layers.GRU(hidden_size, return_sequences=True, return_state=True, name='decoder_gru')
decoder_out, decoder_state = decoder_gru(decoder_inputs, initial_state=encoder_state)

# Attention layer
attn_layer = keras.layers.Attention(name='attention_layer')
attn_out, attn_states = attn_layer([encoder_out, decoder_out])

# Concat attention input and decoder GRU output
decoder_concat_input = keras.layers.Concatenate(axis=-1, name='concat_layer')([decoder_out, attn_out])

# Dense layer
dense = keras.layers.Dense(decoder_inputs.shape[1], activation='softmax', name='softmax_layer')
dense_time = keras.layers.TimeDistributed(dense, name='time_distributed_layer')
decoder_pred = dense_time(decoder_concat_input)

# Full model
full_model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_pred)
full_model.compile(optimizer='adam', loss='categorical_crossentropy')

full_model.summary()

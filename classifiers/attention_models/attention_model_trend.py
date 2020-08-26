import keras

__author__ = "Surayez Rahman"

# Attention code taken from:
# https://www.tensorflow.org/api_docs/python/tf/keras/layers/Attention


def build_model(input_shape):

    nb_classes = 2
    output_shape = (40, 2)

    n_feature_maps = 64
    print(input_shape)

    # Variable-length int sequences.
    query_input = keras.Input(input_shape)
    value_input = keras.Input(output_shape)

    print(query_input)
    print(value_input)

    # CNN layer # Use 'same' padding so outputs have the same shape as inputs.

    # Query encoding of shape [batch_size, Tq, filters].
    query_seq_encoding = keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')(query_input)
    # Value encoding of shape [batch_size, Tv, filters].
    value_seq_encoding = keras.layers.Conv1D(filters=100, kernel_size=4, padding='same')(value_input)

    print(query_seq_encoding)
    print(value_seq_encoding)

    # Query-value attention of shape [batch_size, Tq, filters].
    query_value_attention_seq = keras.layers.Attention()([query_seq_encoding, value_seq_encoding])

    print(query_value_attention_seq)


    # Reduce over the sequence axis to produce encodings of shape
    # [batch_size, filters].
    query_encoding = keras.layers.GlobalAveragePooling1D()(query_seq_encoding)
    query_value_attention = keras.layers.GlobalAveragePooling1D()(query_value_attention_seq)

    print(query_value_attention)

    # Concatenate query and document encodings to produce a DNN input layer.
    input_layer = keras.layers.Concatenate()([query_encoding, query_value_attention])

    print(input_layer)

    # gap_layer = keras.layers.GlobalAveragePooling1D()(input_layer)

    output_layer = keras.layers.Dense(nb_classes, activation='softmax')(input_layer)

    print(output_layer)

    model = keras.models.Model(inputs=(query_input, value_input), outputs=output_layer)
    model.summary()

    return model


if __name__ == "__main__":
    input_shape = (40, 266)
    attn_model = build_model(input_shape)

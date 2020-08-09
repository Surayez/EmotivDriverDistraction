import time
import keras
from classifiers.classifiers import predict_model_deep_learning
from utils.tools import save_logs

__author__ = "Chang Wei Tan"


# Most of the code here are taken from https://github.com/hfawaz/dl-4-tsc
# Attention model explanation: https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39
# Attention code: https://github.com/thushv89/attention_keras

class Classifier_Attention_1:
    def __init__(self, output_directory, input_shape, verbose=False):
        if verbose:
            print('[ResNet] Creating Attention Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        self.model = self.build_model(input_shape)
        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape):

        n_hidden = 100
        output_shape = (2)

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
        decoder_input = keras.layers.RepeatVector(output_train.shape[1])(encoder_last_h)
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
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        model.summary()

        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val=None, yimg_val=None):
        if self.verbose:
            print('[Attention] Training Attention Classifier')
        epochs = 3
        batch_size = 64
        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.h5'
        if Ximg_val is not None:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                          min_lr=0.0001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                               save_best_only=True)
        else:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                          min_lr=0.0001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                               save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        start_time = time.time()
        # train the model
        if Ximg_val is not None:
            self.hist = self.model.fit(Ximg_train, yimg_train,
                                       validation_data=(Ximg_val, yimg_val),
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       batch_size=mini_batch_size,
                                       callbacks=self.callbacks)
        else:
            self.hist = self.model.fit(Ximg_train, yimg_train,
                                       verbose=self.verbose,
                                       epochs=epochs,
                                       batch_size=mini_batch_size,
                                       callbacks=self.callbacks)

        self.duration = time.time() - start_time

        if self.verbose:
            print('[ResNet] Training done!, took {}s'.format(self.duration))

    def predict(self, Ximg, yimg):
        if self.verbose:
            print('[ResNet] Predicting')

        model = keras.models.load_model(self.output_directory + 'best_model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model_deep_learning(model, Ximg, yimg, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()

        if self.verbose:
            print('[ResNet] Prediction done!')

        return model_metrics, conf_mat

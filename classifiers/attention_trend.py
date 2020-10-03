import time

import keras

from classifiers.classifiers import predict_model_deep_learning
from tensorflow.python.keras.layers import GRU, Concatenate
from utils.tools import save_logs

__author__ = "Surayez Rahman"


# Most of the code here are taken from https://github.com/hfawaz/dl-4-tsc
# Attention model explanation: https://towardsdatascience.com/light-on-math-ml-attention-with-keras-dc8dbc1fad39
# Attention code: https://github.com/thushv89/attention_keras

class Classifier_Attention_Trend:
    def __init__(self, output_directory, input_shape, epoch, verbose=False):
        if verbose:
            print('[AttentionTrend] Creating Attention Trend Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        self.model = self.build_model(input_shape)
        self.epoch = epoch
        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape):

        n_hidden = 100
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

        model = keras.attention_models.Model(inputs=input_train, outputs=out)
        opt = keras.optimizers.Adam(lr=0.01, clipnorm=1)
        model.compile(loss='mean_squared_error', optimizer=opt, metrics=['mae'])
        model.summary()

        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val=None, yimg_val=None):
        if self.verbose:
            print('[AttentionTrend] Training ResNet Classifier')
        epochs = self.epoch
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
            print('[AttentionTrend] Training done!, took {}s'.format(self.duration))

    def predict(self, Ximg, yimg):
        if self.verbose:
            print('[AttentionTrend] Predicting')

        model = keras.attention_models.load_model(self.output_directory + 'best_model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model_deep_learning(model, Ximg, yimg, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()

        if self.verbose:
            print('[AttentionTrend] Prediction done!')

        return model_metrics, conf_mat

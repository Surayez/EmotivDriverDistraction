import time

import keras
from keras import Sequential, Model
from keras.layers import Conv1D, Dropout, GlobalAveragePooling1D, Dense, TimeDistributed, \
    Flatten, LSTM, BatchNormalization, Activation

from classifiers.classifiers import predict_model_deep_learning
from utils.classifier_tools import create_class_weight
from utils.tools import save_logs

__author__ = "Chang Wei Tan"


class Classifier_FCN_LSTM:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False, dropout_rate=0.3):
        if verbose:
            print('[FCN-LSTM] Creating FCN-LSTM Classifier')

        self.verbose = verbose
        self.dropout_rate = dropout_rate
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes):

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

        gap_layer = keras.layers.pooling.GlobalAveragePooling1D()(conv3)

        cnn_model = TimeDistributed(Model(inputs=input_layer, outputs=gap_layer))(main_input)

        lstm_layer = LSTM(64, return_sequences=True)(cnn_model)
        lstm_layer = LSTM(64)(lstm_layer)

        output_layer = Dense(64, activation='relu')(lstm_layer)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=main_input, outputs=output_layer)

        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val, yimg_val):
        if self.verbose:
            print('[FCN-LSTM] Training FCN-LSTM Classifier')
        epochs = 2000
        batch_size = 16
        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))

        # compile the model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)

        file_path = self.output_directory + 'best_model.h5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        start_time = time.time()
        # train the model
        self.hist = self.model.fit(Ximg_train, yimg_train,
                                   validation_data=(Ximg_val, yimg_val),
                                   verbose=self.verbose,
                                   epochs=epochs,
                                   batch_size=mini_batch_size,
                                   callbacks=self.callbacks)
        self.duration = time.time() - start_time

        if self.verbose:
            print('[FCN-LSTM] Training done!, took {}s'.format(self.duration))

    def predict(self, Ximg, yimg):
        if self.verbose:
            print('[FCN-LSTM] Predicting')
        model = keras.models.load_model(self.output_directory + 'best_model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model_deep_learning(model, Ximg, yimg, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()
        if self.verbose:
            print('[FCN-LSTM] Prediction done!')

        return model_metrics, conf_mat

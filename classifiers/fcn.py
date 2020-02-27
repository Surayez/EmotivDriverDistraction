import time

import keras

from classifiers.classifiers import predict_model_deep_learning
from utils.classifier_tools import create_class_weight
from utils.tools import save_logs

__author__ = "Chang Wei Tan"


# Most of the code here are taken from https://github.com/hfawaz/dl-4-tsc

class Classifier_FCN:
    def __init__(self, output_directory, input_shape, nb_classes, verbose=False):
        if verbose:
            print('Creating FCN Classifier')
        self.verbose = verbose
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes):
        input_layer = keras.layers.Input(input_shape)

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

        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        # return model
        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val, yimg_val):
        if self.verbose:
            print('[FCN] Training FCN Classifier')
        epochs = 2000
        batch_size = 16
        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))

        self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
                                                      min_lr=0.0001)
        file_path = self.output_directory + 'best_model.h5'
        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss',
                                                           save_best_only=True)
        self.callbacks = [reduce_lr, model_checkpoint]

        # create class weights based on the y label proportions for each image
        class_weight = create_class_weight(yimg_train)

        start_time = time.time()
        # train the model
        self.hist = self.model.fit(Ximg_train, yimg_train,
                                   validation_data=(Ximg_val, yimg_val),
                                   class_weight=class_weight,
                                   verbose=self.verbose,
                                   epochs=epochs,
                                   batch_size=mini_batch_size,
                                   callbacks=self.callbacks)

        self.duration = time.time() - start_time

        if self.verbose:
            print('[FCN] Training done!, took {}s'.format(self.duration))

    def predict(self, Ximg, yimg):
        if self.verbose:
            print('[FCN] Predicting')

        model = keras.models.load_model(self.output_directory + 'best_model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model_deep_learning(model, Ximg, yimg, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()

        if self.verbose:
            print('[FCN] Prediction done!')

        return model_metrics, conf_mat

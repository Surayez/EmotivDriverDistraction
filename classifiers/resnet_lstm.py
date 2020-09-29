import time

import tensorflow.python.keras as keras
from focal_loss import BinaryFocalLoss
from tensorflow.python.keras import Model
from tensorflow.python.keras.layers import Dropout, Dense, TimeDistributed, Flatten, LSTM, Input, GRU, Concatenate

# from classifiers.attention_decoder import AttentionDecoder
from classifiers.classifiers import predict_model_deep_learning
from utils.classifier_tools import create_class_weight
from utils.tools import save_logs

__author__ = "Chang Wei Tan"


class Classifier_ResNet_LSTM:

    def __init__(self, output_directory, input_shape, nb_classes, epoch, verbose=False, dropout_rate=0.3):
        if verbose:
            print('[ResNet-LSTM] Creating ResNet-LSTM Classifier')
        self.verbose = verbose
        self.window_len = 40
        self.dropout_rate = dropout_rate
        self.output_directory = output_directory
        self.model = self.build_model(input_shape, nb_classes)
        self.epoch = epoch
        if verbose == True:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def build_model(self, input_shape, nb_classes):
        n_feature_maps = 64

        main_input = keras.layers.Input(input_shape)
        input_layer = keras.layers.Input((input_shape[1], input_shape[2]))


        # add model layers
        # BLOCK 1
        conv_x = keras.layers.Conv1D(filters=n_feature_maps,
                                     kernel_size=8,
                                     padding='same')(input_layer)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps,
                                     kernel_size=5,
                                     padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps,
                                     kernel_size=3,
                                     padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps,
                                         kernel_size=1,
                                         padding='same')(input_layer)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        # BLOCK 2
        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                     kernel_size=8,
                                     padding='same')(output_block_1)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                     kernel_size=5,
                                     padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                     kernel_size=3,
                                     padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                         kernel_size=1,
                                         padding='same')(output_block_1)
        shortcut_y = keras.layers.normalization.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                     kernel_size=8,
                                     padding='same')(output_block_2)
        conv_x = keras.layers.normalization.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                     kernel_size=5,
                                     padding='same')(conv_x)
        conv_y = keras.layers.normalization.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv1D(filters=n_feature_maps * 2,
                                     kernel_size=3,
                                     padding='same')(conv_y)
        conv_z = keras.layers.normalization.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.normalization.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)
        print(output_block_3)

        # FINAL
        gap_layer = keras.layers.GlobalAveragePooling1D()(output_block_3)
        print(gap_layer)

        cnn_model = TimeDistributed(Model(inputs=input_layer, outputs=gap_layer))(main_input)
        print(cnn_model)

        lstm_layer = LSTM(n_feature_maps, return_sequences=True)(cnn_model)
        lstm_layer = LSTM(n_feature_maps)(lstm_layer)

        output_layer = Dense(n_feature_maps, activation='relu')(lstm_layer)
        output_layer = keras.layers.Dense(nb_classes, activation='softmax')(output_layer)

        model = keras.models.Model(inputs=main_input, outputs=output_layer)
        model.summary()

        return model

    def fit(self, Ximg_train, yimg_train, Ximg_val, yimg_val):
        if self.verbose:
            print('[ResNet-LSTM] Training ResNet-LSTM Classifier')

        epochs = self.epoch
        batch_size = 16
        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))

        METRICS = [
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

        # compile the model
        self.model.compile(loss=BinaryFocalLoss(gamma=2), optimizer='adam', metrics=METRICS)

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
            print('[ResNet-LSTM] Training done!, took {}s'.format(self.duration))

    def predict(self, Ximg, yimg):
        if self.verbose:
            print('[ResNet-LSTM] Predicting')
        model = keras.models.load_model(self.output_directory + 'best_model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model_deep_learning(model, Ximg, yimg, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()
        if self.verbose:
            print('[ResNet-LSTM] Prediction done!')

        return model_metrics, conf_mat

import time
import keras
from classifiers.classifiers import predict_model_deep_learning
from utils.tools import save_logs
from classifiers.attention_models import attention_model, attention_model_fcn, attention_model_resnet

__author__ = "Chang Wei Tan and Surayez Rahman"


class Classifier_Attention:
    def __init__(self, output_directory, input_shape, verbose=False):
        if verbose:
            print('[Attention] Creating Attention Classifier')
        self.verbose = verbose
        self.output_directory = output_directory

        # UPDATE the following line to use desired model
        self.model = attention_model_fcn.build_model(input_shape)

        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

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
            print('[Attention] Training done!, took {}s'.format(self.duration))

    def predict(self, Ximg, yimg):
        if self.verbose:
            print('[Attention] Predicting')

        model = keras.models.load_model(self.output_directory + 'best_model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model_deep_learning(model, Ximg, yimg, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()

        if self.verbose:
            print('[Attention] Prediction done!')

        return model_metrics, conf_mat

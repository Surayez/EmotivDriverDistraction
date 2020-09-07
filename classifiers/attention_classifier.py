import time
import keras
from classifiers.classifiers import predict_model_deep_learning
from utils.tools import save_logs
from classifiers.attention_models import attention_model, attention_model_fcn, attention_model_resnet, \
    attention_experiment, attention_model_bidirectional, multiheadattention_model, selfattention_fcn, \
    selfattention_resnet, multiheadattention_fcn, multiheadattention_resnet

from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention

__author__ = "Chang Wei Tan & Surayez Rahman"


class Classifier_Attention:
    def __init__(self, classifier_name, output_directory, input_shape, epoch, verbose=False):
        self.epoch = epoch

        self.classifier_name = classifier_name
        if verbose:
            print('[' + self.classifier_name + '] Creating Attention Classifier')
        self.verbose = verbose
        self.output_directory = output_directory

        # UPDATE the following line to use desired model
        if classifier_name == "attention_bidirectional":
            self.model = attention_model_bidirectional.build_model(input_shape)
        elif classifier_name == "attention_resnet":
            self.model = attention_model_resnet.build_model(input_shape)
        elif classifier_name == "attention_fcn":
            self.model = attention_model_fcn.build_model(input_shape)
        elif classifier_name == "MHA":
            self.model = multiheadattention_model.build_model(input_shape)
        elif classifier_name == "SA_FCN":
            self.model = selfattention_fcn.build_model(input_shape)
        elif classifier_name == "SA_ResNet":
            self.model = selfattention_resnet.build_model(input_shape)
        elif classifier_name == "MHA_FCN":
            self.model = multiheadattention_fcn.build_model(input_shape)
        elif classifier_name == "MHA_ResNet":
            self.model = multiheadattention_resnet.build_model(input_shape)
        else:
            self.model = attention_experiment.build_model(input_shape)

        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def fit(self, Ximg_train, yimg_train, Ximg_val=None, yimg_val=None):

        if self.verbose:
            print('[' + self.classifier_name + '] Training Attention Classifier')
        epochs = self.epoch
        batch_size = 16
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
            print('[' + self.classifier_name + '] Training done!, took {}s'.format(self.duration))

    def predict(self, Ximg, yimg):
        if self.verbose:
            print('[' + self.classifier_name + '] Predicting')

        if ("SA" in self.classifier_name):
            model = keras.models.load_model(self.output_directory + 'best_model.h5',
                                            custom_objects={'SeqSelfAttention': SeqSelfAttention})
        elif ("MHA" in self.classifier_name or "experiment" in self.classifier_name):
            model = keras.models.load_model(self.output_directory + 'best_model.h5',
                                            custom_objects={'MultiHeadAttention': MultiHeadAttention})
        else:
            model = keras.models.load_model(self.output_directory + 'best_model.h5')

        model_metrics, conf_mat, y_true, y_pred = predict_model_deep_learning(model, Ximg, yimg, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()

        if self.verbose:
            print('[' + self.classifier_name + '] Prediction done!')

        return model_metrics, conf_mat

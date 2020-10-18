import time
import keras
from classifiers.classifiers import predict_model_deep_learning
from focal_loss import BinaryFocalLoss
from utils.tools import save_logs
from classifiers.attention.attention_singular_models import MHSA_ResNet, MHSA_FCN, MHSA, LSA
from classifiers.attention.attention_models import ATTN_experiment, LATTN_ResNet, \
    MHA_ResNet, SelfA_ResNet, MHA_FCN, MHA, SelfA_FCN_Multiplicative, \
    LATTN_FCN, BATTN_ResNet, SelfA_FCN, LATTN, SelfA, BATTN_FCN, SelfA_FCN_Global

from keras_self_attention import SeqSelfAttention
from keras_multi_head import MultiHeadAttention
from classifiers.attention.attention_implements.PreProcessingLayer import PreProcessingLayer

__author__ = "Surayez Rahman"


class Classifier_Attention:
    def __init__(self, classifier_name, output_directory, input_shape, epoch, verbose=False):
        self.epoch = epoch

        self.classifier_name = classifier_name
        if verbose:
            print('[' + self.classifier_name + '] Creating Attention Classifier')
        self.verbose = verbose
        self.output_directory = output_directory

        # UPDATE the following line to use desired model
        if classifier_name == "LATTN":
            self.model = LATTN.build_model(input_shape)
        elif classifier_name == "LATTN_ResNet":
            self.model = LATTN_ResNet.build_model(input_shape)
        elif classifier_name == "LATTN_FCN":
            self.model = LATTN_FCN.build_model(input_shape)
        elif classifier_name == "BATTN_ResNet":
            self.model = BATTN_ResNet.build_model(input_shape)
        elif classifier_name == "BATTN_FCN":
            self.model = BATTN_FCN.build_model(input_shape)
        elif classifier_name == "MHA":
            self.model = MHA.build_model(input_shape)
        elif classifier_name == "LSA":
            self.model = LSA.build_model(input_shape)
        elif classifier_name == "MHSA":
            self.model = MHSA.build_model(input_shape)
        elif classifier_name == "MHSA_ResNet":
            self.model = MHSA_ResNet.build_model(input_shape)
        elif classifier_name == "MHSA_FCN":
            self.model = MHSA_FCN.build_model(input_shape)
        elif classifier_name == "SelfA":
            self.model = SelfA.build_model(input_shape)
        elif classifier_name == "SelfA_FCN":
            self.model = SelfA_FCN.build_model(input_shape)
        elif classifier_name == "SelfA_FCN_Global":
            self.model = SelfA_FCN_Global.build_model(input_shape)
        elif classifier_name == "SelfA_FCN_Multiplicative":
            self.model = SelfA_FCN_Multiplicative.build_model(input_shape)
        elif classifier_name == "SelfA_ResNet":
            self.model = SelfA_ResNet.build_model(input_shape)
        elif classifier_name == "MHA_FCN":
            self.model = MHA_FCN.build_model(input_shape)
        elif classifier_name == "MHA_ResNet":
            self.model = MHA_ResNet.build_model(input_shape)
        else:
            self.model = ATTN_experiment.build_model(input_shape)

        if verbose:
            self.model.summary()

        self.model.save_weights(self.output_directory + 'model_init.h5')

    def fit(self, Ximg_train, yimg_train, Ximg_val=None, yimg_val=None):

        METRICS = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            # keras.metrics.Precision(name='precision'),
            # keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc'),
        ]

        if self.verbose:
            print('[' + self.classifier_name + '] Training Attention Classifier')
        epochs = self.epoch
        batch_size = 16
        mini_batch_size = int(min(Ximg_train.shape[0] / 10, batch_size))

        if "ATTN" in self.classifier_name or "LSA" in self.classifier_name or "experiment" in self.classifier_name:
            self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        else:
            self.model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=METRICS)

        file_path = self.output_directory + 'best_model.h5'
        if Ximg_val is not None:
            # https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/ModelCheckpoint
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=50,
                                                          min_lr=0.0001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='val_loss', save_best_only=True)
        else:
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50,
                                                          min_lr=0.0001)
            model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='accuracy', mode='max', save_best_only=True)
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

        model = keras.models.load_model(self.output_directory + 'best_model.h5',
                                        custom_objects={'MultiHeadAttention': MultiHeadAttention,
                                                        'SeqSelfAttention': SeqSelfAttention,
                                                        # 'PreProcessingLayer': PreProcessingLayer
                                                        })

        model_metrics, conf_mat, y_true, y_pred = predict_model_deep_learning(model, Ximg, yimg, self.output_directory)
        save_logs(self.output_directory, self.hist, y_pred, y_true, self.duration)

        keras.backend.clear_session()

        if self.verbose:
            print('[' + self.classifier_name + '] Prediction done!')

        return model_metrics, conf_mat

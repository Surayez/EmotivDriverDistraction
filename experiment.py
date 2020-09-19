import os
import sys

import numpy as np
import pandas as pd

from utils import data_loader
from utils.classifier_tools import prepare_inputs_deep_learning, prepare_inputs, prepare_inputs_cnn_lstm
from utils.tools import create_directory

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

__author__ = "Chang Wei Tan"


def attention_models():
    return ["attention", "SelfA", "MHA", "SA"]


def lstm_models():
    return ["lstm", "attention", "SelfA", "MHA"]


def get_window_stride(problem):
    window_len = 20
    stride = 1
    binary = True
    if problem == "Emotiv266":
        window_len = 40
        stride = 20
    elif problem == "EmotivRaw":
        window_len = 256
        stride = 128
    elif problem == "ActivityRecognition":
        window_len = 52
        stride = 21
        binary = False
    elif problem == "FordChallenge":
        window_len = 20
        stride = 10
        binary = False
    elif problem == "EEGEyeState":
        window_len = 50
        stride = 25
        binary = False

    return window_len, stride, binary


def fit_classifier(all_labels, X_train, y_train, X_val=None, y_val=None):
    nb_classes = len(np.unique(all_labels))

    if any(x in classifier_name for x in lstm_models()):
        input_shape = (None, X_train.shape[2], X_train.shape[3])
    else:
        input_shape = (X_train.shape[1], X_train.shape[2])

    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory)
    if X_val is None:
        classifier.fit(X_train, y_train)
    else:
        classifier.fit(X_train, y_train, X_val, y_val)

    return classifier


def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=True):
    if any(x in classifier_name for x in attention_models()):
        from classifiers import attention_classifier
        return attention_classifier.Classifier_Attention(classifier_name, output_directory, input_shape, epoch=1, verbose=True)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_ResNet(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "fcn":
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "cnn":
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "cnn_lstm":
        from classifiers import cnn_lstm
        return cnn_lstm.Classifier_CNN_LSTM(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "fcn_lstm":
        from classifiers import fcn_lstm
        return fcn_lstm.Classifier_FCN_LSTM(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "resnet_lstm":
        from classifiers import resnet_lstm
        return resnet_lstm.Classifier_ResNet_LSTM(output_directory, input_shape, nb_classes, verbose)

    if classifier_name == "rocket":
        from classifiers import rocket
        return rocket.Classifier_Rocket(output_directory, input_shape, nb_classes, verbose)


def run_experiments():
    print("#########################################################################")
    print("[Main] Run Experiments")
    print("#########################################################################")
    print("[Main] Data path: {}".format(data_path))
    print("[Main] Output Dir: {}".format(output_directory))
    print("[Main] Problem: {}".format(problem))
    print("[Main] Classifier: {}".format(classifier_name))
    print("[Main] Iteration: {}".format(itr))
    window_len, stride, binary = get_window_stride(problem=problem)
    print("[Main] Window Len: {}".format(window_len))
    print("[Main] Stride: {}".format(stride))
    print("#########################################################################")

    data_folder = data_path + problem + "/"
    train_file = data_folder + problem + "_TRAIN.csv"
    test_file = data_folder + problem + "_TEST.csv"

    train_data = data_loader.load_segmentation_data(train_file)
    test_data = data_loader.load_segmentation_data(test_file)
    print("[Main] {} train series".format(len(train_data)))
    print("[Main] {} test series".format(len(test_data)))

    if classifier_name == "rocket":
        X_train, y_train, X_test, y_test = prepare_inputs(train_inputs=train_data,
                                                          test_inputs=test_data,
                                                          window_len=window_len,
                                                          stride=stride,
                                                          binary=binary)
        print("[Main] Train series:", X_train.shape)
        print("[Main] Test series", X_test.shape)

        all_labels = np.concatenate((y_train, y_test), axis=0)
        print("[Main] All labels: {}".format(np.unique(all_labels)))

        classifier = fit_classifier(all_labels, X_train, y_train)

        metrics_train, _ = classifier.predict(X_train, y_train)
        metrics_test, conf_mat = classifier.predict(X_test, y_test)

        metrics_train['train/val/test'] = 'train'
        metrics_test['train/val/test'] = 'test'

        metrics = pd.concat([metrics_train, metrics_test]).reset_index(drop=True)
        print(metrics.head())
    else:
        if any(x in classifier_name for x in lstm_models()):
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_inputs_cnn_lstm(train_inputs=train_data,
                                                                                     test_inputs=test_data,
                                                                                     window_len=window_len,
                                                                                     stride=stride,
                                                                                     binary=binary)
        else:
            X_train, y_train, X_val, y_val, X_test, y_test = prepare_inputs_deep_learning(train_inputs=train_data,
                                                                                          test_inputs=test_data,
                                                                                          window_len=window_len,
                                                                                          stride=stride,
                                                                                          binary=binary)

        print("[Main] Train series:", X_train.shape)
        if X_val is not None:
            print("[Main] Val series:", X_val.shape)
        print("[Main] Test series", X_test.shape)

        if y_val is not None:
            all_labels = np.concatenate((y_train, y_val, y_test), axis=0)
        else:
            all_labels = np.concatenate((y_train, y_test), axis=0)
        print("[Main] All labels: {}".format(np.unique(all_labels)))

        tmp = pd.get_dummies(all_labels).values

        y_train = tmp[:len(y_train)]
        y_val = tmp[len(y_train):len(y_train) + len(y_val)]
        y_test = tmp[len(y_train) + len(y_val):]

        classifier = fit_classifier(all_labels, X_train, y_train, X_val, y_val)

        metrics_train, _ = classifier.predict(X_train, y_train)
        metrics_val, _ = classifier.predict(X_val, y_val)
        metrics_test, conf_mat = classifier.predict(X_test, y_test)

        metrics_train['train/val/test'] = 'train'
        metrics_val['train/val/test'] = 'val'
        metrics_test['train/val/test'] = 'test'

        metrics = pd.concat([metrics_train, metrics_val, metrics_test]).reset_index(drop=True)

        print(metrics.head())

    metrics.to_csv(output_directory + 'classification_metrics.csv')
    np.savetxt(output_directory + 'confusion_matrix.csv', conf_mat, delimiter=",")


if len(sys.argv) >= 6:
    data_path = sys.argv[1]
    output_directory = sys.argv[2]
    problem = sys.argv[3]
    classifier_name = sys.argv[4]
    itr = sys.argv[5]
else:
    cwd = os.getcwd()
    data_path = cwd + "/TS_Segmentation/"
    output_directory = cwd + "/output/"
    problem = "Emotiv266"
    classifier_name = "MHA"
    itr = "itr_0"

output_directory = output_directory + classifier_name + '/' + problem + '/' + itr + '/'
create_directory(output_directory)

run_experiments()

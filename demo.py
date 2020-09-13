import sys
import os

import numpy as np
import pandas as pd

from utils import data_loader
from utils.classifier_tools import prepare_inputs_deep_learning, prepare_inputs, prepare_inputs_cnn_lstm, \
    prepare_inputs_attention
from utils.tools import create_directory

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

__author__ = "Chang Wei Tan"


def fit_classifier(epoch, all_labels, X_train, y_train, X_val=None, y_val=None):
    nb_classes = len(np.unique(all_labels))

    if (classifier_name == "fcn_lstm") or (classifier_name == "resnet_lstm") or ("attention" in classifier_name):
        input_shape = (None, X_train.shape[2], X_train.shape[3])
    else:
        input_shape = (X_train.shape[1], X_train.shape[2])


    classifier = create_classifier(classifier_name, input_shape, nb_classes, epoch)
    if X_val is None:
        classifier.fit(X_train, y_train)
    else:
        classifier.fit(X_train, y_train, X_val, y_val)

    return classifier


def create_classifier(classifier_name, input_shape, nb_classes, epoch, verbose=True):
    # if classifier_name == 'attention_trend':
    #     from classifiers import attention_trend
    #     return attention_trend.Classifier_Attention_Trend(output_directory, input_shape, verbose)
    if "attention" in classifier_name:
        from classifiers import attention_classifier
        return attention_classifier.Classifier_Attention(classifier_name, output_directory, input_shape, epoch, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_ResNet(output_directory, input_shape, nb_classes, epoch, verbose)
    if classifier_name == "fcn":
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, epoch, verbose)
    if classifier_name == "cnn":
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "cnn_lstm":
        from classifiers import cnn_lstm
        return cnn_lstm.Classifier_CNN_LSTM(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == "fcn_lstm":
        from classifiers import fcn_lstm
        return fcn_lstm.Classifier_FCN_LSTM(output_directory, input_shape, nb_classes, epoch, verbose)
    if classifier_name == "resnet_lstm":
        from classifiers import resnet_lstm
        return resnet_lstm.Classifier_ResNet_LSTM(output_directory, input_shape, nb_classes, epoch, verbose)
    if classifier_name == "rocket":
        from classifiers import rocket
        return rocket.Classifier_Rocket(output_directory, input_shape, nb_classes, verbose)

# todo make this real time and read from Emotiv device
cwd = os.getcwd()
data_path = cwd + "/TS_Segmentation/"
output_directory = cwd + "/output/"
problem = "Emotiv266"
classifier_name = "attention"
window_len = 40
stride = 20
binary = True
epoch = 3

output_directory = output_directory + classifier_name + '/' + problem + '/'
create_directory(output_directory)

print("#########################################################################")
print("[Demo] Run Demo")
print("#########################################################################")
print("[Demo] Data path: {}".format(data_path))
print("[Demo] Output Dir: {}".format(output_directory))
print("[Demo] Problem: {}".format(problem))
print("[Demo] Classifier: {}".format(classifier_name))
print("[Demo] Window Len: {}".format(window_len))
print("[Demo] Stride: {}".format(stride))
print("#########################################################################")

data_folder = data_path + problem + "/"
train_file = data_folder + problem + "_TRAIN.csv"
test_file = data_folder + problem + "_TEST.csv"

train_data = data_loader.load_segmentation_data(train_file)
test_data = data_loader.load_segmentation_data(test_file)
print("[Demo] {} train series".format(len(train_data)))
print("[Demo] {} test series".format(len(test_data)))




if classifier_name == "rocket":
    X_train, y_train, X_test, y_test = prepare_inputs(train_inputs=train_data,
                                                      test_inputs=test_data,
                                                      window_len=window_len,
                                                      stride=stride,
                                                      binary=binary)
    print("[Demo] Train series:", X_train.shape)
    print("[Demo] Test series", X_test.shape)

    all_labels = np.concatenate((y_train, y_test), axis=0)
    print("[Demo] All labels: {}".format(np.unique(all_labels)))

    classifier = fit_classifier(epoch, all_labels, X_train, y_train)

    metrics_train, _ = classifier.predict(X_train, y_train)
    metrics_test, conf_mat = classifier.predict(X_test, y_test)

    metrics_train['train/val/test'] = 'train'
    metrics_test['train/val/test'] = 'test'

    metrics = pd.concat([metrics_train, metrics_test]).reset_index(drop=True)
    print(metrics.head())

else:
    if (classifier_name == "attention_trend"):
        X_train, y_train, X_val, y_val, X_test, y_test = prepare_inputs_attention(train_inputs=train_data,
                                                                                 test_inputs=test_data,
                                                                                 window_len=window_len,
                                                                                 stride=stride,
                                                                                 binary=binary)
    else:
        if (classifier_name == "fcn_lstm") or (classifier_name == "resnet_lstm") or ("attention" in classifier_name):
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

    print("[Demo] Train series:", X_train.shape)
    if X_val is not None:
        print("[Demo] Val series:", X_val.shape)
    print("[Demo] Test series", X_test.shape)

    #
    # if (classifier_name == "attention_trend"):
    #     # attn_classifier = fit_attn_classifier(X_train, y_train, X_val, y_val)
    #
    #     metrics_train, _ = attn_classifier.predict(X_train, y_train)
    #     metrics_val, _ = attn_classifier.predict(X_val, y_val)
    #     metrics_test, conf_mat = attn_classifier.predict(X_test, y_test)
    #
    #     metrics_train['train/val/test'] = 'train'
    #     metrics_val['train/val/test'] = 'val'
    #     metrics_test['train/val/test'] = 'test'
    #
    #     metrics = pd.concat([metrics_train, metrics_val, metrics_test]).reset_index(drop=True)
    #
    #     print(metrics.head())
    #
    #
    # else:
    if y_val is not None:
        all_labels = np.concatenate((y_train, y_val, y_test), axis=0)
    else:
        all_labels = np.concatenate((y_train, y_test), axis=0)
    print("[Demo] All labels: {}".format(np.unique(all_labels)))

    tmp = pd.get_dummies(all_labels).values

    y_train = tmp[:len(y_train)]
    y_val = tmp[len(y_train):len(y_train) + len(y_val)]
    y_test = tmp[len(y_train) + len(y_val):]

    classifier = fit_classifier(epoch, all_labels, X_train, y_train, X_val, y_val)

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

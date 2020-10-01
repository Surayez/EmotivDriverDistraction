import numpy as np
import pandas as pd

import sklearn.metrics as m
from sklearn.metrics import confusion_matrix

__author__ = "Chang Wei Tan"


def get_binary_labels_softmax(y):
    # get the driving labels as a binary 1D array
    y_1 = np.asarray([1 * (y[i][1] > y[i][0]) for i in range(len(y))])
    # get the distracted labels as a binary 1D array
    y_0 = abs(y_1 - 1)
    return y_0, y_1


def get_binary_labels(y):
    # get the driving labels as a binary 1D array
    y_1 = y
    # get the distracted labels as a binary 1D array
    y_0 = abs(y_1 - 1)

    return y_0, y_1


def get_predictions_from_softmax(y):
    return np.argmax(y, axis=1)


def predict_model_deep_learning(model, X, y, output_dir=''):
    if ("Emotiv" in output_dir) and (y.shape[1] == 2):
        return predict_model_deep_learning_emotiv_binary(model, X, y)
    # make a prediction with the model
    yhat = model.predict(X)

    ytrue = get_predictions_from_softmax(y)
    yhat = get_predictions_from_softmax(yhat)

    result_dic = {"accuracy": m.accuracy_score(ytrue, yhat)}
    conf_mat = confusion_matrix(ytrue, yhat)
    print(conf_mat)

    # put the model predictions into a dataframe
    model_metrics = pd.DataFrame.from_dict(result_dic)

    # calculate summary stats
    return model_metrics, conf_mat, ytrue, yhat


def predict_model_deep_learning_emotiv_binary(model, X, y):
    # make a prediction with the model
    yhat = model.predict(X)

    result_dic = {}
    result_dic["accuracy"] = []
    result_dic["precision_distracted"] = []
    result_dic["recall_distracted"] = []
    result_dic["f1_score_distracted"] = []
    result_dic["precision_driving"] = []
    result_dic["recall_driving"] = []
    result_dic["f1_score_driving"] = []
    result_dic["auc_distracted"] = []
    result_dic["auc_driving"] = []

    y_true_distr, y_true_drive = get_binary_labels_softmax(y)

    yhat_distr, yhat_drive = get_binary_labels_softmax(yhat)

    # calculate metrics
    result_dic["accuracy"].append(m.accuracy_score(y_true_distr, yhat_distr))
    result_dic["precision_distracted"].append(m.precision_score(y_true_distr, yhat_distr))
    result_dic["recall_distracted"].append(m.recall_score(y_true_distr, yhat_distr))
    result_dic["f1_score_distracted"].append(m.f1_score(y_true_distr, yhat_distr))
    result_dic["precision_driving"].append(m.precision_score(y_true_drive, yhat_drive))
    result_dic["recall_driving"].append(m.recall_score(y_true_drive, yhat_drive))
    result_dic["f1_score_driving"].append(m.f1_score(y_true_drive, yhat_drive))
    result_dic["auc_distracted"].append(m.roc_auc_score(y_true_distr, yhat_distr))
    result_dic["auc_driving"].append(m.roc_auc_score(y_true_drive, yhat_drive))

    conf_mat = confusion_matrix(y_true_drive, yhat_drive)
    print(conf_mat)
    # put the model predictions into a dataframe
    model_metrics = pd.DataFrame.from_dict(result_dic)

    # calculate summary stats
    return model_metrics, conf_mat, y_true_drive, yhat_drive


def predict_model(model, X, y, output_dir=''):
    if ("Emotiv" in output_dir) and (len(np.unique(y)) == 2):
        # if dataset is Emotiv and binary class
        return predict_model_emotiv_binary(model, X, y)
    # make a prediction with the model
    yhat = model.predict(X)

    result_dic = {"accuracy": m.accuracy_score(y, yhat)}
    conf_mat = confusion_matrix(y, yhat)
    print(conf_mat)

    # put the model predictions into a dataframe
    model_metrics = pd.DataFrame.from_dict(result_dic)

    # calculate summary stats
    return model_metrics, conf_mat, y, yhat


def predict_model_emotiv_binary(model, X, y):
    # make a prediction with the model for Emotiv dataset
    yhat = model.predict(X)

    result_dic = {}
    result_dic["accuracy"] = []
    result_dic["precision_distracted"] = []
    result_dic["recall_distracted"] = []
    result_dic["f1_score_distracted"] = []
    result_dic["precision_driving"] = []
    result_dic["recall_driving"] = []
    result_dic["f1_score_driving"] = []
    result_dic["auc_distracted"] = []
    result_dic["auc_driving"] = []

    y_true_distr, y_true_drive = get_binary_labels(y)
    yhat_distr, yhat_drive = get_binary_labels(yhat)

    # calculate metrics
    result_dic["accuracy"].append(m.accuracy_score(y_true_distr, yhat_distr))
    result_dic["precision_distracted"].append(m.precision_score(y_true_distr, yhat_distr))
    result_dic["recall_distracted"].append(m.recall_score(y_true_distr, yhat_distr))
    result_dic["f1_score_distracted"].append(m.f1_score(y_true_distr, yhat_distr))
    result_dic["precision_driving"].append(m.precision_score(y_true_drive, yhat_drive))
    result_dic["recall_driving"].append(m.recall_score(y_true_drive, yhat_drive))
    result_dic["f1_score_driving"].append(m.f1_score(y_true_drive, yhat_drive))
    result_dic["auc_distracted"].append(m.roc_auc_score(y_true_distr, yhat_distr))
    result_dic["auc_driving"].append(m.roc_auc_score(y_true_drive, yhat_drive))

    conf_mat = confusion_matrix(y, yhat)
    print(conf_mat)

    # put the model predictions into a dataframe
    model_metrics = pd.DataFrame.from_dict(result_dic)

    # calculate summary stats
    return model_metrics, conf_mat, y_true_drive, yhat_drive

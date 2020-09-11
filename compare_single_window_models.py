import sys
import getopt
import csv
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from utils import data_loader
from utils.classifier_tools import prepare_inputs, prepare_inputs_cnn_lstm, prepare_inputs_deep_learning
from utils.tools import create_directory

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

__author__ = "Chang Wei Tan and Surayez Rahman"


def results_table(classifier_names, train, test, val):
    models = ["Models"] + classifier_names
    result_train = ["Train"] + train
    result_test = ["Test"] + test
    result_val = ["Val"] + val

    # Create a results table CSV
    table = open("results_table.csv", "w", newline="")
    writer = csv.writer(table, delimiter=',')
    writer.writerows([models, result_train, result_test, result_val])
    table.close()


def graph_label(rects):
    # Ref: https://matplotlib.org/3.2.1/gallery/lines_bars_and_markers/barchart.html
    for rect in rects:
        height = rect.get_height()
        plt.annotate('{}'.format(height),
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 3 points vertical offset
                     textcoords="offset points",
                     ha='center', va='bottom')


def results_chart(classifier_names, train, test, val):
    train = list(np.around(np.array(train), 2))
    test = list(np.around(np.array(test), 2))
    val = list(np.around(np.array(val), 2))

    N = len(classifier_names)
    ind = np.arange(N)
    width = 0.25

    rects1 = plt.bar(ind, train, width, label='Train')
    rects2 = plt.bar(ind + width, val, width, label='Val')
    rects3 = plt.bar(ind + width * 2, test, width, label='Test')

    plt.ylabel('Scores')
    plt.title('Scores by Train/Val/Test')

    plt.xticks(ind + width / 2, classifier_names)
    plt.legend(loc='best')

    graph_label(rects1)
    graph_label(rects2)
    graph_label(rects3)

    plt.savefig("result_bar.png")
    plt.show()
    plt.close()


def fit_classifier(classifier_name, epoch, output_directory, all_labels, x_train, y_train, X_val=None, y_val=None):
    nb_classes = len(np.unique(all_labels))

    # if (classifier_name == "fcn_lstm") or (classifier_name == "resnet_lstm") or ("attention" in classifier_name):
    if any(x in classifier_name for x in ["fcn_lstm", "resnet_lstm", "attention", "SA", "MHA"]):
        input_shape = (None, x_train.shape[2], x_train.shape[3])
    else:
        input_shape = (x_train.shape[1], x_train.shape[2])

    classifier = create_classifier(classifier_name, output_directory, input_shape, nb_classes, epoch)
    if X_val is None:
        classifier.fit(x_train, y_train)
    else:
        classifier.fit(x_train, y_train, X_val, y_val)

    return classifier


def create_classifier(classifier_name, output_directory, input_shape, nb_classes, epoch, verbose=True):
    if any(x in classifier_name for x in ["attention", "SA", "MHA"]):
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


def run_rocket(data, epoch, window_len, stride, binary):
    train_data = data[1]
    test_data = data[2]
    output_directory = data[7]

    X_train, y_train, X_test, y_test = prepare_inputs(train_inputs=train_data,
                                                      test_inputs=test_data,
                                                      window_len=window_len,
                                                      stride=stride,
                                                      binary=binary)
    print("[Compare_Models] Train series:", X_train.shape)
    print("[Compare_Models] Test series", X_test.shape)

    all_labels = np.concatenate((y_train, y_test), axis=0)
    print("[Compare_Models] All labels: {}".format(np.unique(all_labels)))

    classifier = fit_classifier("rocket", epoch, output_directory, all_labels, X_train, y_train)

    metrics_train, _ = classifier.predict(X_train, y_train)
    metrics_test, conf_mat = classifier.predict(X_test, y_test)

    metrics_train['train/val/test'] = 'train'
    metrics_test['train/val/test'] = 'test'

    metrics = pd.concat([metrics_train, metrics_test]).reset_index(drop=True)
    return metrics, conf_mat


def run_deep_learning_models(classifier_name, data, epoch):
    all_labels = data[0]
    X_train = data[1]
    y_train = data[2]
    X_val = data[3]
    y_val = data[4]
    X_test = data[5]
    y_test = data[6]
    output_directory = data[7]

    # Fit the classifier
    classifier = fit_classifier(classifier_name, epoch, output_directory, all_labels, X_train, y_train, X_val, y_val)

    # Predict using classifier [Train, Val, Test]
    metrics_train, _ = classifier.predict(X_train, y_train)
    metrics_val, _ = classifier.predict(X_val, y_val)
    metrics_test, conf_mat = classifier.predict(X_test, y_test)

    metrics_train['train/val/test'] = 'train'
    metrics_val['train/val/test'] = 'val'
    metrics_test['train/val/test'] = 'test'

    metrics = pd.concat([metrics_train, metrics_val, metrics_test]).reset_index(drop=True)
    return metrics, conf_mat


def run_model(classifier_name, data, epoch, window_len, stride, binary):
    output_directory = data[7]
    if classifier_name == "rocket":
        metrics, conf_mat = run_rocket(data, epoch, window_len, stride, binary)

    else:
        metrics, conf_mat = run_deep_learning_models(classifier_name, data, epoch)

    metrics.to_csv(output_directory + 'classification_metrics.csv')
    np.savetxt(output_directory + 'confusion_matrix.csv', conf_mat, delimiter=",")

    return metrics


def prepare_data_cnn_lstm(problem, window_len, stride, binary, data_version):
    # Set up output location
    cwd = os.getcwd()
    data_path = cwd + "/TS_Segmentation/"
    output_directory = cwd + "/output/"
    output_directory = output_directory + "compare_models" + '/' + problem + '/'
    create_directory(output_directory)

    print("#########################################################################")
    print("[Compare_Models] Run Compare_Models")
    print("#########################################################################")
    print("[Compare_Models] Data path: {}".format(data_path))
    print("[Compare_Models] Output Dir: {}".format(output_directory))
    print("[Compare_Models] Problem: {}".format(problem))
    print("[Compare_Models] Window Len: {}".format(window_len))
    print("[Compare_Models] Stride: {}".format(stride))
    print("#########################################################################")

    data_folder = data_path + problem + "/"
    train_file = data_folder + problem + "_TRAIN.csv"
    test_file = data_folder + problem + "_TEST.csv"

    train_data = data_loader.load_segmentation_data(train_file)
    test_data = data_loader.load_segmentation_data(test_file)
    print("[Compare_Models] {} train series".format(len(train_data)))
    print("[Compare_Models] {} test series".format(len(test_data)))

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_inputs_cnn_lstm(train_inputs=train_data,
                                                                             test_inputs=test_data,
                                                                             window_len=window_len,
                                                                             stride=stride,
                                                                             binary=binary,
                                                                             data_version=data_version)

    X_train, y_train, X_val, y_val, X_test, y_test = prepare_inputs_deep_learning(train_inputs=train_data,
                                                                                  test_inputs=test_data,
                                                                                  window_len=window_len,
                                                                                  stride=stride,
                                                                                  binary=binary)

    print("[Compare_Models] Train series:", X_train.shape)
    if X_val is not None:
        print("[Compare_Models] Val series:", X_val.shape)
    print("[Compare_Models] Test series", X_test.shape)

    if y_val is not None:
        all_labels = np.concatenate((y_train, y_val, y_test), axis=0)
    else:
        all_labels = np.concatenate((y_train, y_test), axis=0)
    print("[Compare_Models] All labels: {}".format(np.unique(all_labels)))

    tmp = pd.get_dummies(all_labels).values

    y_train = tmp[:len(y_train)]
    y_val = tmp[len(y_train):len(y_train) + len(y_val)]
    y_test = tmp[len(y_train) + len(y_val):]

    return [all_labels, X_train, y_train, X_val, y_val, X_test, y_test, output_directory]


def main(argv):
    problem = "Emotiv266"
    classifier_names = ["MHA"]
    epoch = 5
    window_len = 40
    stride = 20
    data_version = ""
    # # For EmotivRaw:
    # window_len = 256
    # stride = 128
    binary = True

    # Command line args
    try:
        opts, args = getopt.getopt(argv, "p:c:e:", ["problem=", "classifier=", "epoch="])
    except getopt.GetoptError:
        print("Incorrect arguments passed")
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-p", "--problem"):
            problem = arg
        elif opt in ("-c", "--classifier"):
            classifier = arg
            classifier_names = classifier.split(',')
        elif opt in ("-e", "--epoch"):
            epoch = arg

    # Print runtime information
    print("Problem:", problem, "\nClassifiers:", classifier_names, "\nEpochs:", epoch)

    # Prepare results arrays
    result_train = []
    result_test = []
    result_val = []

    # Prepare Data
    data = prepare_data_cnn_lstm(problem, window_len, stride, binary, data_version)

    for classifier_name in classifier_names:
        # Run each Model
        metrics = run_model(classifier_name, data, epoch, window_len, stride, binary)
        print(metrics.head())

        result_train.append(metrics["accuracy"].values[0] * 100)
        result_val.append(metrics["accuracy"].values[1] * 100)
        result_test.append(metrics["accuracy"].values[2] * 100)

    results_table(classifier_names, result_train, result_test, result_val)
    results_chart(classifier_names, result_train, result_test, result_val)


if __name__ == "__main__":
    main(sys.argv[1:])
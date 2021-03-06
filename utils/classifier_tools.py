import math

import numpy as np
from collections import Counter
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

# Most of the code here are taken from https://github.com/hfawaz/dl-4-tsc
__author__ = "Chang Wei Tan and Surayez Rahman"


def up_sample(x_train, y_train):
    # Ref: https://stackoverflow.com/questions/56125380/resampling-data-using-smote-from-imblearn-with-3d-numpy-arrays
    print('Original dataset shape {}'.format(x_train.shape))
    print('Original dataset classes {}'.format(Counter(y_train)))
    orig_shape = x_train.shape
    x_train_reshaped = np.reshape(x_train, (x_train.shape[0], x_train.shape[1] * x_train.shape[2]))

    ros = RandomOverSampler(random_state=0)
    X_res, y_res = ros.fit_sample(x_train_reshaped, y_train)
    X_res_shape = X_res.shape

    x_train_final = np.reshape(X_res, (X_res_shape[0], orig_shape[1], orig_shape[2]))
    print('Re-sampled dataset shape {}'.format(x_train_final.shape))
    print('Re-sampled dataset classes {}'.format(Counter(y_res)))

    return x_train_final, y_res


def prepare_inputs_attention(train_inputs, test_inputs, window_len=40, stride=20,
                             val_size=2, random_state=1234, binary=True, class_one=None, verbose=1):
    # This function prepare the inputs to have the right shape for attention attention_models.
    # The shape we are after is (n_series, series_len, series_dim)
    # Inputs are df with data and label columns
    # Inputs:
    #   train_inputs:   training dataset
    #   test_inputs:    test dataset
    #   window_len:     subsequence window size
    #   stride:         stride
    #   val_size:       number of series to be used as validation
    #   random_state:   random seed
    #   binary:         whether we convert to binary case
    #   class_one:      the classes to be used as class one
    #   verbose:        verbosity

    if class_one is None:
        class_one = [3, 11]
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    if len(train_inputs) > val_size:
        train_series, val_series = train_test_split([x for x in range(len(train_inputs))],
                                                    test_size=val_size,
                                                    random_state=random_state)
    else:
        train_series = range(len(train_inputs))
        val_series = None

    X_train = []
    y_train = []
    for i in train_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences_attention(this_series, this_series_labels,
                                                                 window_size=window_len,
                                                                 stride=stride,
                                                                 binary=binary,
                                                                 class_one=class_one)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if val_series is None:
        X_val = None
        y_val = None
    else:
        X_val = []
        y_val = []
        for i in val_series:
            this_series = train_inputs.data[i]
            this_series_labels = train_inputs.label[i]
            subsequences, sub_label = extract_subsequences_attention(this_series, this_series_labels,
                                                                     window_size=window_len,
                                                                     stride=stride,
                                                                     binary=binary,
                                                                     class_one=class_one)
            [X_val.append(x) for x in subsequences]
            [y_val.append(x) for x in sub_label]
        X_val = np.array(X_val)
        y_val = np.array(y_val)

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences_attention(this_series, this_series_labels,
                                                                 window_size=window_len,
                                                                 stride=stride,
                                                                 binary=binary,
                                                                 class_one=class_one)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_inputs_deep_learning(train_inputs, test_inputs, window_len=40, stride=20,
                                 val_size=2, random_state=1234, binary=True, class_one=None, verbose=1, data_version="", norm=True):
    # This function prepare the inputs to have the right shape for deep learning attention_models.
    # The shape we are after is (n_series, series_len, series_dim)
    # Inputs are df with data and label columns
    # Inputs:
    #   train_inputs:   training dataset
    #   test_inputs:    test dataset
    #   window_len:     subsequence window size
    #   stride:         stride
    #   val_size:       number of series to be used as validation
    #   random_state:   random seed
    #   binary:         whether we convert to binary case
    #   class_one:      the classes to be used as class one
    #   verbose:        verbosity
    if class_one is None:
        class_one = [1, 2, 3, 11]
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    if len(train_inputs) > val_size:
        train_series, val_series = train_test_split([x for x in range(len(train_inputs))],
                                                    test_size=val_size,
                                                    random_state=random_state)
    else:
        train_series = range(len(train_inputs))
        val_series = None

    X_train = []
    y_train = []
    for i in train_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        if data_version == "enhanced":
            subsequences, sub_label = extract_enhanced_subsequences(this_series, this_series_labels,
                                                                    window_size=window_len,
                                                                    stride=stride,
                                                                    binary=binary,
                                                                    class_one=class_one)
        else:
            subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                           window_size=window_len,
                                                           stride=stride,
                                                           binary=binary,
                                                           class_one=class_one,
                                                           norm=norm)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    if val_series is None:
        X_val = None
        y_val = None
    else:
        X_val = []
        y_val = []
        for i in val_series:
            this_series = train_inputs.data[i]
            this_series_labels = train_inputs.label[i]
            subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                           window_size=window_len,
                                                           stride=stride,
                                                           binary=binary,
                                                           class_one=class_one,
                                                           norm=norm)
            [X_val.append(x) for x in subsequences]
            [y_val.append(x) for x in sub_label]
        X_val = np.array(X_val)
        y_val = np.array(y_val)

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride,
                                                       binary=binary,
                                                       class_one=class_one,
                                                       norm=norm)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_inputs_cnn_lstm(train_inputs, test_inputs, window_len=40, stride=20,
                            val_size=2, random_state=1234, n_subs=4, binary=True, class_one=None, verbose=1,
                            data_version=""):
    # This function prepare the inputs to have the right shape for deep learning attention_models specifically CNN-LSTM attention_models.
    # The idea is to get n_subs subsequences of length=window_len, pass each of them to a CNN for features and
    # learn the relationship with past subsequences using LSTM.
    # The shape we are after is (n_series, n_subs, series_len, series_dim)
    # Inputs are df with data and label columns
    # Inputs:
    #   train_inputs:   training dataset
    #   test_inputs:    test dataset
    #   window_len:     subsequence window size
    #   stride:         stride
    #   val_size:       number of series to be used as validation
    #   random_state:   random seed
    #   n_subs:         number of subsequences used to learn the relationship
    #   binary:         whether we convert to binary case
    #   class_one:      the classes to be used as class one
    #   verbose:        verbosity

    if class_one is None:
        class_one = [1, 2, 3, 11]
    n_length = window_len
    larger_window = window_len * n_subs
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    train_series, val_series = train_test_split([x for x in range(len(train_inputs))],
                                                test_size=val_size,
                                                random_state=random_state)
    X_train = []
    y_train = []
    for i in train_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        if (data_version == "trimmed"):
            subsequences, sub_label = extract_trimmed_subsequences(this_series, this_series_labels,
                                                                   window_size=larger_window,
                                                                   stride=stride,
                                                                   binary=binary,
                                                                   class_one=class_one)
        elif (data_version == "enhanced"):
            subsequences, sub_label = extract_enhanced_subsequences(this_series, this_series_labels,
                                                                    window_size=window_len,
                                                                    stride=stride,
                                                                    binary=binary,
                                                                    class_one=class_one)
        else:
            subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                           window_size=larger_window,
                                                           stride=stride,
                                                           binary=binary,
                                                           class_one=class_one)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Up-sampling data
    X_train, y_train = up_sample(X_train, y_train)

    X_train = X_train.reshape((X_train.shape[0], n_subs, n_length, X_train.shape[2]))

    X_val = []
    y_val = []
    for i in val_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=larger_window,
                                                       stride=stride,
                                                       binary=binary,
                                                       class_one=class_one)
        [X_val.append(x) for x in subsequences]
        [y_val.append(x) for x in sub_label]
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    X_val = X_val.reshape((X_val.shape[0], n_subs, n_length, X_val.shape[2]))

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=larger_window,
                                                       stride=stride,
                                                       binary=binary,
                                                       class_one=class_one)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_test = X_test.reshape((X_test.shape[0], n_subs, n_length, X_test.shape[2]))

    return X_train, y_train, X_val, y_val, X_test, y_test


def prepare_inputs_combined(train_inputs, test_inputs, window_len=40, stride=20,
                            val_size=2, random_state=1234, n_subs=4, binary=True, class_one=None, verbose=1,
                            data_version="", upsampled=True, norm=True):
    # This function prepare the inputs to have the right shape for deep learning attention_models specifically CNN-LSTM attention_models.
    # The idea is to get n_subs subsequences of length=window_len, pass each of them to a CNN for features and
    # learn the relationship with past subsequences using LSTM.
    # The shape we are after is (n_series, n_subs, series_len, series_dim)
    # Inputs are df with data and label columns
    # Inputs:
    #   train_inputs:   training dataset
    #   test_inputs:    test dataset
    #   window_len:     subsequence window size
    #   stride:         stride
    #   val_size:       number of series to be used as validation
    #   random_state:   random seed
    #   n_subs:         number of subsequences used to learn the relationship
    #   binary:         whether we convert to binary case
    #   class_one:      the classes to be used as class one
    #   verbose:        verbosity

    if class_one is None:
        class_one = [1, 2, 3, 11]
    n_length = window_len
    larger_window = window_len * n_subs
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    train_series, val_series = train_test_split([x for x in range(len(train_inputs))],
                                                test_size=val_size,
                                                random_state=random_state)
    X_train = []
    y_train = []
    for i in train_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        # if (data_version == "trimmed"):
        #     subsequences, sub_label = extract_trimmed_subsequences(this_series, this_series_labels,
        #                                                            window_size=larger_window,
        #                                                            stride=stride,
        #                                                            binary=binary,
        #                                                            class_one=class_one)
        # elif (data_version == "enhanced"):
        #     subsequences, sub_label = extract_enhanced_subsequences(this_series, this_series_labels,
        #                                                             window_size=larger_window,
        #                                                             stride=stride,
        #                                                             binary=binary,
        #                                                             class_one=class_one)
        # else:
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=larger_window,
                                                       stride=stride,
                                                       binary=binary,
                                                       class_one=class_one,
                                                       norm=norm
                                                       )
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Up-sampling data
    if upsampled == True:
        X_train, y_train = up_sample(X_train, y_train)

    X_train = X_train.reshape((X_train.shape[0], n_subs, n_length, X_train.shape[2]))

    X_val = []
    y_val = []
    for i in val_series:
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=larger_window,
                                                       stride=stride,
                                                       binary=binary,
                                                       class_one=class_one,
                                                       norm=norm)
        [X_val.append(x) for x in subsequences]
        [y_val.append(x) for x in sub_label]
    X_val = np.array(X_val)
    y_val = np.array(y_val)
    X_val = X_val.reshape((X_val.shape[0], n_subs, n_length, X_val.shape[2]))

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=larger_window,
                                                       stride=stride,
                                                       binary=binary,
                                                       class_one=class_one,
                                                       norm = norm)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    X_test = X_test.reshape((X_test.shape[0], n_subs, n_length, X_test.shape[2]))

    dataset1 = [X_train, y_train, X_val, y_val, X_test, y_test]

    X2_train, y2_train, X2_val, y2_val, X2_test, y2_test = prepare_inputs_deep_learning(train_inputs=train_inputs,
                                                                                 test_inputs=test_inputs,
                                                                                 window_len=window_len,
                                                                                 stride=stride,
                                                                                 binary=binary, data_version=data_version,
                                                                                 norm=norm)

    # Up-sampling data
    if upsampled == True:
        X2_train, y2_train = up_sample(X2_train, y2_train)

    dataset2 = [X2_train, y2_train, X2_val, y2_val, X2_test, y2_test]

    return dataset1, dataset2


def prepare_inputs(train_inputs, test_inputs, window_len=40, stride=20, binary=True, class_one=None, verbose=1):
    # This function prepare the inputs to have the right shape for ML attention_models without validation set.
    # The shape we are after is (n_series, series_len, series_dim)
    # Inputs are df with data and label columns
    # Inputs:
    #   train_inputs:   training dataset
    #   test_inputs:    test dataset
    #   window_len:     subsequence window size
    #   stride:         stride
    #   binary:         whether we convert to binary case
    #   class_one:      the classes to be used as class one
    #   verbose:        verbosity
    if class_one is None:
        class_one = [3, 11]
    if verbose > 0:
        print('[ClassifierTools] Preparing inputs')

    X_train = []
    y_train = []
    for i in range(len(train_inputs)):
        this_series = train_inputs.data[i]
        this_series_labels = train_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride,
                                                       binary=binary,
                                                       class_one=class_one)
        [X_train.append(x) for x in subsequences]
        [y_train.append(x) for x in sub_label]
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    X_test = []
    y_test = []
    for i in range(len(test_inputs)):
        this_series = test_inputs.data[i]
        this_series_labels = test_inputs.label[i]
        subsequences, sub_label = extract_subsequences(this_series, this_series_labels,
                                                       window_size=window_len,
                                                       stride=stride,
                                                       binary=binary,
                                                       class_one=class_one)
        [X_test.append(x) for x in subsequences]
        [y_test.append(x) for x in sub_label]
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    return X_train, y_train, X_test, y_test


def extract_subsequences_attention(X_data, y_data, window_size=30, stride=1, binary=True, class_one=None, norm=True):
    # This function extract subsequences from a long time series for Attention
    # Assumes that each timestamp has a laextract_subsequencesbel represented by y_data.
    # The label for each subsequence is taken with the majority class in that segment.
    if class_one is None:
        class_one = [3, 11]
    data_len, data_dim = X_data.shape

    subsequences = []
    next_subsequences = []
    count = 0
    for i in range(0, data_len, stride):
        end = i + window_size
        if end + window_size > data_len:
            break
        tmp = X_data[i:end, :]
        tmp2 = X_data[i + window_size:end + window_size, :]

        if norm:
            # usually z-normalisation is required for TSC
            scaler = StandardScaler()
            tmp = scaler.fit_transform(tmp)
            tmp2 = scaler.fit_transform(tmp2)
        subsequences.append(tmp)
        next_subsequences.append(tmp2)

        count += 1

    return np.array(subsequences), np.array(next_subsequences)


def extract_trimmed_subsequences(X_data, y_data, window_size=30, stride=1, binary=True, class_one=None, norm=True):
    # This function extract subsequences from a long time series.
    # Assumes that each timestamp has a label represented by y_data.
    # The label for each subsequence is taken with the majority class in that segment.
    if class_one is None:
        class_one = [3, 11]
    data_len, data_dim = X_data.shape

    subsequences = []
    labels = []
    count = 0
    for i in range(0, data_len, stride):
        end = i + window_size
        if end > data_len:
            break
        consistent_labels = checkSame(y_data[i:end])
        if consistent_labels:
            label = stats.mode(y_data[i:end]).mode[0]
            # label = stats.mode(y_data[i:end]).mode[0]
            tmp = X_data[i:end, :]
            if norm:
                # usually z-normalisation is required for TSC
                scaler = StandardScaler()
                tmp = scaler.fit_transform(tmp)
            subsequences.append(tmp)
            if binary:
                label = make_binary(label, class_one=class_one)
            labels.append(label)

            count += 1
    return np.array(subsequences), np.array(labels)


def extract_enhanced_subsequences(X_data, y_data, window_size=40, stride=1, binary=True, class_one=None, norm=True):
    # This function extract subsequences from a long time series.
    # Assumes that each timestamp has a label represented by y_data.
    # The label for each subsequence is taken with the majority class in that segment.
    if class_one is None:
        class_one = [3, 11]
    data_len, data_dim = X_data.shape

    subsequences = []
    labels = []
    count = 0
    for i in range(0, data_len, stride):
        end = i + window_size
        if end > data_len:
            break

        tmp = X_data[i:end, :]
        if norm:
            # usually z-normalisation is required for TSC
            scaler = StandardScaler()
            tmp = scaler.fit_transform(tmp)

        tmp, label = fix_block(x_data=tmp, y_data=y_data[i:end], class_one=class_one, block_size=window_size)
        if tmp is not None:
            subsequences.append(tmp)
            if binary:
                label = make_binary(label, class_one=class_one)
            labels.append(label)

        count += 1

    return np.array(subsequences), np.array(labels)


def check_block(x_data, y_data, data_len, class_one):
    block_size = 40
    location = 0
    while location + block_size <= data_len:
        block_y = y_data[location: location + block_size]
        cutOff = calculate_cutoff(block_y, class_one)
        if cutOff is not None:
            additional = block_size - cutOff
            for i in range(additional):
                data_to_add = x_data[location + cutOff - 1]
                y_data_to_add = y_data[location + cutOff - 1]
                x_data = np.insert(x_data, location + cutOff, data_to_add, 0)
                y_data = np.insert(y_data, location + cutOff, y_data_to_add, 0)
        else:
            location += block_size

    return x_data, y_data


def fix_block(x_data, y_data, class_one, block_size):
    cutOff = calculate_cutoff(y_data, class_one)

    if cutOff is not None:
        if cutOff >= block_size / 2:
            additional = block_size - cutOff
            x_processed_data = x_data[:cutOff, ]
            y_processed_data = y_data[:cutOff, ]
            label = stats.mode(y_data[:cutOff]).mode[0]
            for i in range(additional):
                x_data_to_add = x_data[cutOff - 1]
                y_data_to_add = y_data[cutOff - 1]
                x_processed_data = np.insert(x_processed_data, cutOff + i, x_data_to_add, 0)
                y_processed_data = np.insert(y_processed_data, cutOff + i, y_data_to_add, 0)
        else:
            x_processed_data = x_data[cutOff:, ]
            y_processed_data = y_data[cutOff:, ]
            label = stats.mode(y_data[cutOff:]).mode[0]
            for i in range(cutOff):
                x_data_to_add = x_data[block_size - 1]
                y_data_to_add = y_data[block_size - 1]
                x_processed_data = np.insert(x_processed_data, 0, x_data_to_add, 0)
                y_processed_data = np.insert(y_processed_data, 0, y_data_to_add, 0)

        cutOff = calculate_cutoff(y_processed_data, class_one)
        if cutOff is not None:
            print("Enhancement skipped")
            return None, None
        return x_processed_data, label

    else:
        return x_data, y_data[0]


def calculate_cutoff(ydata, class_one):
    exists_class_one = False
    y_label = ydata[0]
    if y_label in class_one:
        exists_class_one = True
    count = 0
    for i in range(1, len(ydata) - 1):
        count += 1
        if exists_class_one and ydata[i] not in class_one:
            return count
        elif exists_class_one is False and ydata[i] in class_one:
            return count
    return None


# def fix_window(x_data, y_data, window_size, binary, class_one):
#     tmp_labels = []
#     tmp_dataAll = []
#     while not checkSame(y_data):
#         cutOffPnt = calculateCutOff(y_data)
#         tmp_data = x_data[:cutOffPnt]
#         while len(tmp_data) < window_size:
#             tmp_data = np.concatenate((tmp_data, tmp_data))
#
#         tmp_label = y_data[0]
#         if binary:
#             tmp_label = make_binary(tmp_label, class_one=class_one)
#
#         tmp_labels.append(tmp_label)
#         tmp_dataAll.append(tmp_data[:160])
#         y_data = y_data[cutOffPnt:]
#         x_data = x_data[cutOffPnt:]
#
#     if len(y_data) != 0:
#         tmp_data = x_data[:]
#         while len(tmp_data) < window_size:
#             tmp_data = np.concatenate((tmp_data, tmp_data))
#         tmp_label = y_data[0]
#         if binary:
#             tmp_label = make_binary(tmp_label, class_one=class_one)
#         tmp_labels.append(tmp_label)
#         tmp_dataAll.append(tmp_data[:160])
#
#     return tmp_dataAll, tmp_labels


def checkSame(iterator, class_one):
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(first == rest for rest in iterator)


def extract_subsequences(X_data, y_data, window_size=30, stride=1, binary=True, class_one=None, norm=True):
    # This function extract subsequences from a long time series.
    # Assumes that each timestamp has a label represented by y_data.
    # The label for each subsequence is taken with the majority class in that segment.
    if class_one is None:
        class_one = [3, 11]
    data_len, data_dim = X_data.shape

    subsequences = []
    labels = []
    count = 0
    for i in range(0, data_len, stride):
        end = i + window_size
        if end > data_len:
            break
        tmp = X_data[i:end, :]
        if norm:
            # usually z-normalisation is required for TSC
            scaler = StandardScaler()
            tmp = scaler.fit_transform(tmp)
        subsequences.append(tmp)
        label = stats.mode(y_data[i:end]).mode[0]
        if binary:
            label = make_binary(label, class_one=class_one)
        labels.append(label)

        count += 1

    return np.array(subsequences), np.array(labels)


def make_binary(y, class_one):
    if y in class_one:
        return 1
    else:
        return 0


def create_class_weight(labels, mu=2):
    labels_dict = {}
    total = 0
    for i in range(labels.shape[1]):
        labels_dict.update({i: sum(labels[:, i])})
        total += sum(labels[:, i])

    keys = labels_dict.keys()
    class_weight = dict()

    for key in keys:
        score = math.log(mu * total / float(labels_dict[key]))
        class_weight[key] = score if score > 1.0 else 1.0

    return class_weight

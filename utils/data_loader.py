import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

__author__ = "Chang Wei Tan"


def load_segmentation_data(filepath, norm=False, verbose=1):
    # Load the data from csv. Assumes that the data is in dataframe format.
    # Data has columns for:
    #   series:     the id for each series in the dataset
    #   label:      the running labels for every timepoint
    #   timestamp:  timestamp of the data
    #   d1-dN:      time series data with N dimensions
    #
    # Read the data in that format and store it into a new format with data and label columns.
    # The data is an array of shape = seq_len, n_dim
    if verbose > 0:
        print("[SegmentationDataLoader] Loading data from {}".format(filepath))
    df = pd.read_csv(filepath)
    all_series = df.series.unique()

    data = []
    for series in all_series:
        if verbose > 0:
            print("[SegmentationDataLoader] Processing series {}".format(series))

        this_series = df.loc[df.series == series].reset_index(drop=True)
        series_labels = np.array(this_series.label)
        series_data = np.array(this_series.iloc[:, 3:])
        if norm:
            scaler = StandardScaler()
            series_data = scaler.fit_transform(series_data)
        data.append(pd.DataFrame({"data": [series_data],
                                  "label": [series_labels]}, index=[0]))
    data = pd.concat(data)
    data.reset_index(drop=True, inplace=True)

    return data

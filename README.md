# EmotivDriverDistraction
This repository provides the source code for a IJCAI 2020 demo on detecting driver's distraction.

## How to run the code?
There are two main files that you can run in this repository. 
* demo.py runs the demo for the detecting distraction. The user has to specify the classifier, dataset (including path), subsequence window length and stride in the file.
* experiment.py is used to run experiments that can be called from terminal. The command line arguments are data_path, output_directory, problem, classifier_name, iteration.

## Classifiers implemented in this demo
1. RandOm Convolutional KErnel Transform (Rocket), https://github.com/angus924/rocket. The original code was only for univariate TSC and was modified by the authors for multivariate TSC.
2. Fully Convolutional Network (FCN), from https://github.com/hfawaz/dl-4-tsc/
3. Residual Network (ResNet), from https://github.com/hfawaz/dl-4-tsc/
4. FCN-LSTM, our proposed network using FCN as the feature extraction layer and trained a LSTM to learn the relationship between subsequences.
5. ResNet-LSTM, our proposed network using ResNet as the feature extraction layer and trained a LSTM to learn the relationship between subsequences.


Note that the dataset is not included in this repository.
Please contact me if you would like to get a hold of the dataset for testing purposes. 

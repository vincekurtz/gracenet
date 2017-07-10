#!/usr/bin/env python3

##
#
# GraceNET v0.0
#
# Predict future anomolies based soley on the past 24 months of
# GRACE anomolies. This file creates and trains a neural network 
# to predict future changes in GRACE based on past changes.
#
##

# Multi-Layer Perceptron Regression system. 
# See http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib  # for saving parameters
import json

train_data = "/home/vince/Groundwater/NeuralNet/training_data.json"

def load_data(datafile):
    """
    Get training (or test) data from plaintext files.
    Return X, y, where X is the 12d previous months input
    and y is the (1d) anomoly.
    """
    print("===> Loading Data")
    with open(datafile, 'r') as f:
        data = json.load(f)
        X = data['X']
        y = data['y']
    return (X, y)

def create_and_train(X, y):
    """
    Create the network with certain parameters.
    then train it with the given data.
    """
    print("===> Creating Network")
    net = MLPRegressor(
            solver='adam',
            alpha=1e-5,
            activation='logistic',
            hidden_layer_sizes=(5,2),
            random_state=1)

    print("===> Training Network")
    net.fit(X, y)

    return net


if __name__=="__main__":
    X, y = load_data("training_data.json")
    rgr = create_and_train(X, y)

    # save the weights/parameters
    joblib.dump(rgr, 'parameters.pkl')




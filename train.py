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
from sin_approx import get_sinusoid_params

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
            solver='sgd',
            alpha=1e-5,
            activation='logistic',
            hidden_layer_sizes=(50,20),
            random_state=1)

    print("===> Training Network")
    net.fit(X, y)

    return net

def fit_sines(prev_anomolies):
    """
    Calculate sinusoid parameters based on the previous anomolies. 
    We'll use the slope from these to train the network.

    returns a list of lists with the sine parameters
        [[amplitude, phase, mean, slope], ...]
    """
    fits = []
    for i in prev_anomolies:
        fit = get_sinusoid_params(i)
        fits.append(fit)
    return fits

if __name__=="__main__":
    X, y = load_data("training_data.json")
    N = 1000000  # number of samples to train on
    print("Total samples availible = %s" % len(X))
    print("Samples used = %s" % N)

    # just take a subset of availible data for now
    X_in = [X[i] for i in range(N)]
    anomoly = [y[i] for i in range(N)]

    # calculate sinusoid parameters based on the previous anomolies. 
    # we'll use the slope from these to train the network
    #sine_params = fit_sines(prev_anomolies)
    #slopes = [i[3] for i in sine_params]  # the slope is the third parameter

    rgr = create_and_train(X_in, anomoly)

    # save the weights/parameters
    joblib.dump(rgr, 'parameters.pkl')




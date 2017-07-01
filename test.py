#!/usr/bin/env python3

##
#
# GraceNET v0.0
#
# Predict future anomolies based soley on the past 12 months of 
# GRACE anomolies. This script loads a trained neural network and
# runs tests on other data.
#
##

# Multi-Layer Perceptron Regression system. 
# See http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
from sklearn.neural_network import MLPRegressor
from sklearn.externals import joblib
from train import load_data

if __name__=="__main__":
    # load testing data
    X, y = load_data("testing_data.json")
    # load the network
    rgr = joblib.load('parameters.pkl')
    # predict on the network
    pred = rgr.predict(X)
    # get r^2 score
    r2 = rgr.score(X, y)

    print(y)
    print(pred)
    print("")
    print(r2)

    


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
import matplotlib.pyplot as plt
from sin_approx import predict_next  # prediction based on sinusoid only
from sklearn.metrics import r2_score

def plot_stuff(x_example, y_example, predicted):
    """
    Plot a particular data point. Assume that x_example
    contains only the points leading up to y_example.
    """
    plt.plot(range(22), x_example)
    plt.plot(24, y_example, 'og')
    plt.plot(24, predicted, 'xr')
    plt.show()

def predict(x):
    """
    Return a prediction for the next value given 
    a set of variables for a single point.
    """
    prev_grace = x   # later this will change since other variables will be included
    sine_pred = predict_next(x)
    return sine_pred

if __name__=="__main__":
    # load testing data
    X, y = load_data("testing_data.json")

    pred = []
    for x in X:
        pred.append( predict(x) )

    # load the network
    #rgr = joblib.load('parameters.pkl')
    # predict on the network
    #pred = rgr.predict(X)
    # get r^2 score
    #r2 = rgr.score(X, y)

    r2 = r2_score(y, pred)
    
    print(y)
    print(pred)
    print("")
    print(r2)

    #for i in range(2):
    #    plot_stuff(X[i],y[i],pred[i])

    


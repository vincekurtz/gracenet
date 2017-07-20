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
from sklearn.preprocessing import StandardScaler  # for feature scaling
import json
from sin_approx import get_sinusoid_params
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA  # principle component analysis for dim. reduction
import timeit   # timer

train_data = "/home/vince/Groundwater/NeuralNet/training_data.json"

def load_data(datafile):
    """
    Get training (or test) data from plaintext files.
    Return X, y, where X is the 12d previous months input
    and y is the (1d) anomoly.
    """
    with open(datafile, 'r') as f:
        data = json.load(f)
        X = data['X']
        y = data['y']

    X = np.asarray(X)
    y = np.asarray(y)

    return (X, y)   # load as np arrays

def create_and_train(X, y, alpha=1e-5):
    """
    Create the network with certain parameters.  then train it with the given data.
    """
    net = MLPRegressor(
            solver='adam',
            alpha=alpha,
            activation='relu',
<<<<<<< Updated upstream
            hidden_layer_sizes=(2000,1000),
=======
            hidden_layer_sizes=(1000,1000),
>>>>>>> Stashed changes
            random_state=1)

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

def artificial_data(X, y):
    """
    Generate a new row of data with similar
    characteristics, but gaussian noise added so
    that we can get more data points.
    """
    # Generate arrays of random fractions  (~ .9 to 1.1)
    x_rand_perc = 0.1 * np.random.standard_normal(X.shape) + 1  # gaussian with mean 1, std 0.1
    y_rand_perc = 0.1 * np.random.standard_normal(y.shape) + 1

    # Multiply random fractions by original to get new dataset
    X_new = np.multiply(X, x_rand_perc)
    y_new = np.multiply(y, y_rand_perc)

    return (X_new, y_new)

def gen_art_data(X, y, n):
    """
    Add artificial data to the given X and y datasets until
    there are n samples. 
    """
    print("  Total samples availible = %s" % len(y))
    target_samples = n   # how many training examples we'd like to have
    original_X = X
    original_y = y
    while (len(y) < target_samples):
        newx, newy = artificial_data(original_X, original_y)  # only base generated data on the originals
        X = np.concatenate((X, newx))
        y = np.concatenate((y, newy))
        print("    generating artificial data ... %s / %s samples ready" % (len(y), target_samples))
    print("  Total samples with artificial = %s" % len(y))

    return (X, y)

def choose_alpha():
    """
    Make plot of testing and training r^2 values for 
    different values of the regularization parameter.
    """
    test = []
    train = []

    alphas = [3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1, 3, 10]
    for alpha in alphas:
        print("")
        print(" ALPHA = %s" % alpha)
        print("")
        (R_train, R_test) = main(save_data=False, alpha=alpha)
        train.append(R_train)
        test.append(R_test)
    plt.semilogx(alphas, train, 'ro', label='training set')
    plt.semilogx(alphas, test, 'bx', label='test set')
    plt.xlabel('alpha')
    plt.ylabel('R^2 score')
    plt.legend()
    plt.show()

def main(save_data=True, alpha=1e-5):
    """
    Train the network and print basic R^2 scores
    for training and test sets. Then save the parameters
    """

    print("===> Loading Data")
    X, y = load_data("training_data.json")
    X_test, y_test = load_data("testing_data.json")   # for cross validation

    # perform feature scaling
    print("===> Scaling features")
    scaler = StandardScaler()
    scaler.fit(X)  # Don't cheat - fit only on training data
    X = scaler.transform(X)
    X_test = scaler.transform(X_test)   # apply same transformation to test data

    # generate artificial data
    train_samples = 500
    test_samples = 70
    print("===> Generating artificial training data")
    X, y = gen_art_data(X, y, train_samples)
    print("===> Generating artificial testing data")
    X_test, y_test = gen_art_data(X_test, y_test, test_samples)

     dimensionality reduction/principle component analysis
    print("===> Reducing Dimensions")
    print("  old training dimensions (n_samples, n_features): (%s, %s) " % (X.shape[0], X.shape[1]))
    pca = PCA()
    pca.fit(X)
    X = pca.transform(X)  # apply the dim. reduction to both training and test sets
    X_test = pca.transform(X_test)
    print("  new training dimensions (n_samples, n_features): (%s, %s) " % (X.shape[0], X.shape[1]))

    print("===> Creating and Training Network")
    start_time = timeit.default_timer()   # time the training process
    rgr = create_and_train(X, y, alpha)
    end_time = timeit.default_timer()
    print("  %s seconds to train" % (end_time - start_time))
    print("  %s iterations completed" % (rgr.n_iter_))

    print("===> Evaluating Performance")
    training_R = rgr.score(X, y)
    testing_R = rgr.score(X_test, y_test)
    print("")
    print("Training data R^2 score: %s" % training_R)
    print("Testing data R^2 score: %s" % testing_R)
    print("")

    if save_data:  # so you can switch this off sometimes
        print("===> Saving parameters")
        joblib.dump(rgr, 'parameters.pkl')

    return (training_R, testing_R)

if __name__=="__main__":
    choose_alpha()

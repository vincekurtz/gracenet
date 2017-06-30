#!/usr/bin/env python3

##
#
# GraceNET v0.0
#
# Predict future anomolies based soley on the past 12 months of
# GRACE anomolies.
#
##

# Multi-Layer Perceptron Regression system. 
# See http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
from sklearn.neural_network import MLPRegressor
import sys
import random
import math
from matplotlib import pyplot as plt
import numpy as np
import csv
import glob

data_dir = "/home/vince/Groundwater/NeuralNet/data/grace/"


def get_training_data(num_examples):
    """
    Get training data from plaintext files.
    Return X, y, where X is the 12d previous months input
    and y is the (1d) anomoly.
    """
    X = []
    y = []

    for i in range(num_examples):
        # get a good anomoly value
        pixel, year, month, day = random_valid_pixel()
        y.append(get_anomoly(pixel, year, month, day))

        # get all the previous 12 months anomolies for that pixel
        prev = []
        ly, lm, ld = (year, month, day)   # last year month and day
        for j in range(12):
            ly,lm,ld = get_prev_entry(ly,lm,ld)
            anom = get_anomoly(pixel, ly, lm, ld)
            prev.append(anom)


        X.append(prev)
    print(y)
    print(X)


def get_prev_entry(year, month, day):
    """
    For a given month's data, we might like to find the month
    that preceeds it. This function returns the year, month, and
    day that correspond to that month's data.
    """
    files = glob.glob(data_dir + "GRCTellus.JPL*")
    files.sort()   # sorting alphabetically is enough b/c nice naming scheme!
    this_name = data_dir + "GRCTellus.JPL.%04d%02d%02d.LND.RL05_1.DSTvSCS1411.txt" % (year, month, day)
    for i in range(len(files)):
        if files[i] == this_name:
            fname = files[i-1]
            yyyymmdd = fname[-35:-27]  # looking backwards from the end in case data_dir changes
            y = yyyymmdd[0:4]
            m = yyyymmdd[4:6]
            d = yyyymmdd[6:8]
            return (int(y), int(m), int(d))
    return None


def get_anomoly(pixel, year, month, day):
    """
    Return the anomoly found in with the given specifications.
    
    pixel should be a (lon, lat) touple. 
    year, month, and day should be strings.
    """
    fname = data_dir + "GRCTellus.JPL.%04d%02d%02d.LND.RL05_1.DSTvSCS1411.txt" % (year, month, day)
    lon = str(pixel[0])
    lat = str(pixel[1])
    with open(fname, 'r') as fh:
        reader = csv.reader(fh, delimiter=" ")
        for row in reader:
            if (row[0] == lon and row[1] == lat):   # check for matching pixel
                return float(row[2])
    return None

def random_valid_pixel():
    """
    Randomly select a pixel that will yield valid training data.
    This means that the given pixel
        1. Must exist for the given date
        2. Must exist in the previous 12 months

    Return a tuple of pixel, year, month, day
    """
    files = glob.glob(data_dir + "GRCTellus.JPL*")
    files.sort()   # sorting alphabetically is enough b/c nice naming scheme!
    files = files[12:]    # remove the first 12 months since there won't be enough data before these

    startfile = files[random.randint(0,len(files)-1)]  # choose a random month

    # choose a random pixel
    with open(startfile) as f:
        for i, l in enumerate(f):
            pass
    num_lines = i + 1
    header_lines = 21
    pixel_line = random.randint(header_lines, num_lines)

    # get the value of that pixel
    with open(startfile) as f:
        reader = csv.reader(f, delimiter=" ")
        for i, row in enumerate(reader):
            if (i == pixel_line):
                pixel = (float(row[0]), float(row[1]))


    yyyymmdd = startfile[-35:-27]  # looking backwards from the end in case data_dir changes
    year = int(yyyymmdd[0:4])
    month = int(yyyymmdd[4:6])
    day = int(yyyymmdd[6:8])

    # make sure that pixel exists for the previous 12 months
    y, m, d = (year, month, day)
    for i in range(12):
        y, m, d = get_prev_entry(y, m, d)
        if not exists(pixel, y, m, d):
            # one of the previous months doesn't have our given pixel
            # So do we give up? No. We try again
            print("Found invalid pixel. Trying again")
            return random_valid_pixel()

    return (pixel, year, month, day)

    
def exists(pixel, year, month, day):
    """
    Check if a given pixel for a given date exists.
    Return true or false.
    """
    fname = data_dir + "GRCTellus.JPL.%04d%02d%02d.LND.RL05_1.DSTvSCS1411.txt" % (year, month, day)
    lon = str(pixel[0])
    lat = str(pixel[1])
    try:
        with open(fname, 'r') as fh:
            reader = csv.reader(fh, delimiter=" ")
            for row in reader:
                if (row[0] == lon and row[1] == lat):   # check for matching pixel
                    return True   # found the pixel!
        return False
    except:  # if the file can't be opened, it's probably a bad date
        return False



def old():
    """wrapping old stuff in a function for now"""

    # Test data
    xx = np.arange(0,x_range,0.1)
    yy = [math.sin(i) for i in xx]

    # create the neural network
    print("Creating the network")
    rgr = MLPRegressor(solver='lbfgs', alpha=1e-5, activation="tanh",
                        hidden_layer_sizes=(5, 2), random_state=1)

    # train the network
    print("Training the network")
    rgr.fit(X.reshape(-1,1), y)    # need to reshape like this if using only 1d input

    # predict using the network
    print("Running the network on test data")
    pred = rgr.predict(xx.reshape(-1,1))

    print("Plotting")
    plt.plot(X, y, '+g')
    plt.plot(xx,yy)
    plt.plot(xx, pred, '-r')
    plt.show()


if __name__=="__main__":
    get_training_data(3)

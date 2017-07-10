#!/usr/bin/env python3

##
#
# GraceNET v0.0
#
# Predict future anomolies based soley on the past 12 months of
# GRACE anomolies. This file generates training and testing data 
# saving both to json files
#
##

import random
import csv
import glob
import json

data_dir = "/home/vince/Groundwater/NeuralNet/data/grace/"


def get_data(num_examples):
    """
    Get training/testing data from plaintext files.
    Return X, y, where X is the 12d previous months input
    and y is the (1d) anomoly.
    """
    X = []
    y = []

    for i in range(num_examples):
        print("--> Generating sample %s of %s" % (i, num_examples))
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

    return (X, y)

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
            #print("Found invalid pixel. Trying again")
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

if __name__=="__main__":
    n_train = 1000
    n_test = 100

    X, y = get_data(n_train+n_test)

    # Separate training and test sets
    X_train = X[0:n_train]
    y_train = y[0:n_train]
    X_test = X[n_train:]
    y_test = y[n_train:]

    print("\n===> Saving Data to json")
    # save training data in json format
    train_dct = {"y":y_train, "X":X_train}
    with open('training_data.json', 'w') as f:
        json.dump(train_dct, f, indent=2)

    # save testing data in json format
    test_dct = {"y":y_test, "X":X_test}
    with open('testing_data.json', 'w') as f:
        json.dump(test_dct, f, indent=2)

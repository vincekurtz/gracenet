#!/usr/bin/env python3

##
#
# GraceNET v0.0
#
# Predict future anomolies based soley on the past 24 months of
# GRACE anomolies. This file generates training and testing data 
# saving both to json files
#
##

import random
import csv
import glob
import json
import datetime
import numpy as np
import matplotlib.pyplot as plt

data_dir = "/home/vince/Groundwater/NeuralNet/data/"


def get_data(num_examples):
    """
    Get training/testing data from plaintext files.
    Return X, y, where X is the 24d previous months input
    and y is the (1d) anomoly.
    """
    X = []
    y = []

    pixel_list = []  # avoid repeating pixels
    for i in range(num_examples):
        print("--> Generating sample %s of %s" % (i+1, num_examples))
        # get a good pixel
        pixel, year, month, day = random_valid_pixel(pixel_list)
        pixel_list.append(pixel)

        # The target data: a GRACE anomoly
        anom = get_anomoly(pixel, year, month, day)

        # Now deal with input variables for this pixel
        x = [] 

        # get all the previous 24 months anomolies for that pixel
        prev = []
        ly, lm, ld = (year, month, day)   # last year month and day
        for j in range(24):
            ly,lm,ld = get_prev_entry(ly,lm,ld)
            anom = get_anomoly(pixel, ly, lm, ld)
            prev.append(anom)
        x.append(prev)

        # Other variables like vegetation, temperature, etc
        other_vars = []
        other_vars.append(get_veg_trend(pixel, year, month, day))
        other_vars.append(get_temperature_trend(pixel, year, month, day))

        x.append(other_vars)

        if None not in other_vars:
            # a None in other_vars would indicate that there is not enough data for 
            # temperature or vegetation
            X.append(x)
            y.append(anom)

    return (X, y)

def get_prev_entry(year, month, day):
    """
    For a given month's data, we might like to find the month
    that preceeds it. This function returns the year, month, and
    day that correspond to that month's data.
    """
    files = glob.glob(data_dir + "grace/GRCTellus.JPL*")
    files.sort()   # sorting alphabetically is enough b/c nice naming scheme!
    this_name = data_dir + "grace/GRCTellus.JPL.%04d%02d%02d.LND.RL05_1.DSTvSCS1411.txt" % (year, month, day)
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
    fname = data_dir + "grace/GRCTellus.JPL.%04d%02d%02d.LND.RL05_1.DSTvSCS1411.txt" % (year, month, day)
    lon = str(pixel[0])
    lat = str(pixel[1])
    with open(fname, 'r') as fh:
        reader = csv.reader(fh, delimiter=" ")
        for row in reader:
            if (row[0] == lon and row[1] == lat):   # check for matching pixel
                return float(row[2])
    return None

def get_veg_trend(pixel, year, month, day):
    """
    Return the 2 year vegetation trend for a given pixel
    and date. The trend should be over the N months before
    the given date.
    
    pixel should be a (lon, lat) touple. 
    year, month, and day should be strings.
    """
    N = 60
    day_of_year = datetime.datetime(int(year), int(month), int(day)).strftime("%j")
    files = glob.glob(data_dir + "vegetation/MOD13C2_EVI*")
    files.sort()   # sorting alphabetically is enough b/c nice naming scheme!

    # find the vegetation data closest to the requested date
    testf = data_dir + "vegetation/MOD13C2_EVI_%s_%s_monthly.csv" % (year, day_of_year)   # data for the day we'd really like
    if (testf in files):
        # this date is already exactly included!
        startf = testf
    else:
        # we need to look back a bit to find the entry closest to but before
        # the given date
        lst = files + [testf]
        lst.sort()
        for i in range(len(lst)):
            if lst[i] == testf:
                startf = lst[i-1]

    # get data files for the previous N months
    fnames = []
    for i in range(len(files)):
        if files[i] == startf:
            start = i
    for j in range(N):
        fnames.append(files[start-j])

    # get data for this pixel from these previous months
    lon = str(pixel[0])
    lat = str(pixel[1])

    evi = []
    months = []
    n = 0
    for fname in fnames:
        found = False
        with open(fname, 'r') as fh:
            reader = csv.reader(fh, delimiter=" ")
            for row in reader:
                if (row[0] == lon and row[1] == lat):   # check for matching pixel
                    evi.append(float(row[2]))
                    found = True
        if found:
            months.append(n)
        n+=1

    if len(evi) < 10:
        print("no EVI data avilible for this pixel") 
        return None

    # now fit a linear regression of the form y = mx+b
    x = np.array(months)
    y = np.array(evi)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, y_int = np.linalg.lstsq(A, y)[0]

    return(slope)

def get_temperature_trend(pixel, year, month, day):
    """
    Return the 2 year tempearature trend for a given pixel
    and date. The trend should be over the N months before
    the given date.
    
    pixel should be a (lon, lat) touple. 
    year, month, and day should be strings.
    """
    N = 60
    day_of_year = datetime.datetime(int(year), int(month), int(day)).strftime("%j")
    files = glob.glob(data_dir + "temperature/MOD11C3_LST*")
    files.sort()   # sorting alphabetically is enough b/c nice naming scheme!

    # find the vegetation data closest to the requested date
    testf = data_dir + "temperature/MOD11C3_LST_Day_CMG_%s_%s_monthly.csv" % (year, day_of_year)   # data for the day we'd really like
    if (testf in files):
        # this date is already exactly included!
        startf = testf
    else:
        # we need to look back a bit to find the entry closest to but before
        # the given date
        lst = files + [testf]
        lst.sort()
        for i in range(len(lst)):
            if lst[i] == testf:
                startf = lst[i-1]

    # get data files for the previous N months
    fnames = []
    for i in range(len(files)):
        if files[i] == startf:
            start = i
    for j in range(N):
        fnames.append(files[start-j])

    # get data for this pixel from these previous months
    lon = str(pixel[0])
    lat = str(pixel[1])

    temp = []
    months = []
    n = 0
    for fname in fnames:
        found = False
        with open(fname, 'r') as fh:
            reader = csv.reader(fh, delimiter=" ")
            for row in reader:
                if (row[0] == lon and row[1] == lat):   # check for matching pixel
                    temp.append(float(row[2]))
                    found = True
        if found:
            months.append(n)
        n+=1

    if len(temp) < 10:
        print("no temperature data avilible for this pixel") 
        return None

    # now fit a linear regression of the form y = mx+b
    x = np.array(months)
    y = np.array(temp)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, y_int = np.linalg.lstsq(A, y)[0]

    return(slope)

def random_valid_pixel(pixel_list):
    """
    Randomly select a pixel that will yield valid training data.
    This means that the given pixel
        1. Must exist for the given date
        2. Must exist in the previous 24 months
        3. Must not be in pixel_list

    Return a tuple of pixel, year, month, day
    """
    files = glob.glob(data_dir + "grace/GRCTellus.JPL*")
    files.sort()   # sorting alphabetically is enough b/c nice naming scheme!
    files = files[24:]    # remove the first 24 months since there won't be enough data before these

    startfile = files[random.randint(0,len(files)-1)]  # choose a random month

    # choose a random pixel
    with open(startfile) as f:
        for i, l in enumerate(f):
            pass
    num_lines = i
    header_lines = 22
    pixel_line = random.randint(header_lines, num_lines)

    # get the value of that pixel
    with open(startfile) as f:
        reader = csv.reader(f, delimiter=" ")
        for i, row in enumerate(reader):
            if (i == pixel_line):
                pixel = (float(row[0]), float(row[1]))

    # make sure the pixel isn't already in our list
    if pixel in pixel_list:
        # this pixel is already in our list
        #print("pixel already chosen. picking a new one")
        return random_valid_pixel(pixel_list)

    yyyymmdd = startfile[-35:-27]  # looking backwards from the end in case data_dir changes
    year = int(yyyymmdd[0:4])
    month = int(yyyymmdd[4:6])
    day = int(yyyymmdd[6:8])

    # make sure that pixel exists for the previous 24 months
    y, m, d = (year, month, day)
    for i in range(24):
        y, m, d = get_prev_entry(y, m, d)
        if not exists(pixel, y, m, d):
            # one of the previous months doesn't have our given pixel
            # So do we give up? No. We try again
            #print("Found invalid pixel. Trying again")
            return random_valid_pixel(pixel_list)

    return (pixel, year, month, day)

def exists(pixel, year, month, day):
    """
    Check if a given pixel for a given date exists.
    Return true or false.
    """
    fname = data_dir + "grace/GRCTellus.JPL.%04d%02d%02d.LND.RL05_1.DSTvSCS1411.txt" % (year, month, day)
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

def main():
    n_train = 600
    n_test = 10

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

if __name__=="__main__":
    main()
    

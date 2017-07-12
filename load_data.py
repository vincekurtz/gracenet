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
import time
import re

data_dir = "/home/vince/Groundwater/NeuralNet/data/"

grace_data = None
irrigation_data = None
population_data = None
precipitation_data = None
temperature_data = None
vegetation_data = None

def load_all_data():
    """ 
    Load data from files as global variables. 
    Note that this requires a significant amount (~4.5GB)
    of RAM.
    """
    global grace_data
    global irrigation_data
    global population_data
    global precipitation_data
    global temperature_data
    global vegetation_data

    print("===> Loading GRACE data to memory")
    grace_data = get_data_dict('grace/GRC*', 'grace')
    print("===> Loading IRRIGATION data to memory")
    irrigation_data = get_data_dict('grace/GRC*', 'grace')
    print("===> Loading POPULATION data to memory")
    population_data = get_data_dict('grace/GRC*', 'grace')
    print("===> Loading PRECIPITATION data to memory")
    precipitation_data = get_data_dict('grace/GRC*', 'grace')
    print("===> Loading TEMPERATURE data to memory")
    temperature_data = get_data_dict('grace/GRC*', 'grace')
    print("===> Loading VEGETATION data to memory")
    vegetation_data = get_data_dict('grace/GRC*', 'grace')

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
        lat = pixel[1]  # lattitude
        other_vars.append(get_veg_trend(pixel, year, month, day))  # vegetation trend
        other_vars.append(get_temperature_trend(pixel, year, month, day))  # temperature trend
        other_vars.append(lat)
        #other_vars.append(get_irrigation_level(pixel))
        other_vars.append(get_precip_trend(pixel, year, month, day))

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

def get_data_dict(fpattern, fformat):
    """
    Load the data from a given file pattern into a dictionary.
    This dictionary will hold all the data for a given variable. 

    Example input for vegetation: fpattern="vegetation/MOD13C2_EVI*", fformat="veg"
    The fformat is used to differentiate between different file naming conventions.

    Returns: { 
                 DATE: {PIXEL: DATA, PIXEL1: DATA1, ...}, 
                 DATE2: {PIXEL: DATA, PIXEL1: DATA1, ...}, 
               }
    """
    d = {}  # the dictionary that will hold all of our data
    files = glob.glob(data_dir + fpattern)
    files.sort()  # sorting alphabetically puts files in chronological order

    # figure out the date from the filename. 
    # This will depend on the naming conventions of the files, which we learn
    # from the fformat variable.
    if fformat == 'veg':
        # vegetation
        def get_date(fname):
            regex = r'MOD13C2_EVI_([0-9]*)_([0-9]*)_monthly.csv' # year, julian day of year format
            m = re.search(regex, fname)
            year = int(m.group(1))
            day_of_year = int(m.group(2))
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year-1)
            return (date.year, date.month, date.day)
    elif fformat == 'temp':
        # temperature
        def get_date(fname):
            regex = r'MOD11C3_LST_Day_CMG_([0-9]*)_([0-9]*)_monthly.csv'  # year, day of year format
            m = re.search(regex, fname)
            year = int(m.group(1))
            day_of_year = int(m.group(2))
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year-1)
            return (date.year, date.month, date.day)
    elif fformat == 'precip':
        # precipitation
        def get_date(fname):
            regex = r'precipitation_([\.0-9]+)'   # decimal date format
            m = re.search(regex, fname)
            decidate = float(m.group(1))
            year = int(decidate)
            rem = decidate - year
            base = datetime.datetime(year, 1, 1)
            date = base + datetime.timedelta(seconds=(base.replace(year=base.year + 1) - base).total_seconds() * rem)
            return (date.year, date.month, date.day)
    elif fformat == 'pop':
        # population
        def get_date(fname):
            regex = r'population_density_([0-9]*)_regridded.txt'
            m = re.search(regex, fname)
            year = int(m.group(1))
            return (year, 1, 1)   # we only have population data on the year
    elif fformat == 'irr':
        # irritation
        def get_date(fname):
            regex = r'irrigation_pct_([0-9]*).csv'
            m = re.search(regex, fname)
            year = int(m.group(1))
            return (year, 1, 1) 
    elif fformat == 'grace':
        # grace anomoly data
        def get_date(fname):
            regex = r'GRCTellus.JPL.([0-9]*).LND.RL05_1.DSTvSCS1411.txt'
            m = re.search(regex, fname)
            datestring = m.group(1)
            year = int(datestring[0:4])
            month = int(datestring[4:6])
            day = int(datestring[6:8])
            return (year, month, day) 

    else:
        print("ERROR: unrecognized file format %s" % fformat)
        return None

    for fname in files:
        date = get_date(fname)  # (year, month, day)
        data = get_pixel_data(fname)  # {(lon, lat): val, ...}
        
        # add an entry to the dictionary
        d[date] = data

    return d

def get_pixel_data(fname):
    """
    Return a dictionary of pixel tuples (lon, lat) and measurents
    for all lines in the given file. Assume that each file is
    for a unique date, and that columns 0, 1, and 2 are lat, lon,
    and measurement respectively.
    """
    d = {}
    with open(fname, 'r') as fh:
        reader = csv.reader(fh, delimiter=" ")
        for row in reader:
            if row[0] != "HDR":  # exclude header rows
                lon = float(row[0])
                lat = float(row[1])
                meas = float(row[2])
                d[(lon,lat)] = meas
    return d

def get_veg_trend(pixel, year, month, day):
    """
    Return the 2 year vegetation trend for a given pixel
    and date. The trend should be over the N months before
    the given date.
    
    pixel should be a (lon, lat) touple. 
    year, month, and day should be strings.
    """
    N = 24
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

def get_irrigation_level(pixel):
    """
    Get the 2013 percent of land equipped for irrigation for
    a given pixel.
    """
    fname = data_dir + "irrigation/irrigation_pct_2013.csv"
    lon = str(pixel[0])
    lat = str(pixel[1])
    with open(fname, 'r') as fh:
        reader = csv.reader(fh, delimiter=" ")
        for row in reader:
            if (row[0] == lon and row[1] == lat):
                print("irrigation found!")
                return float(row[2])
    # assume no data means the level is zero. This prevents excessive 
    # pruning of pixels, since irrigation data is so sparce. 
    return 0.0  

def get_precip_trend(pixel, year, month, day):
    """
    Return the precipitation trend for a given pixel
    and date. The trend should be over the N months before
    the given date.
    
    pixel should be a (lon, lat) touple. 
    year, month, and day should be strings.
    """
    N = 24
    decidate = str(toYearFraction(datetime.datetime(int(year), int(month), int(day))))[0:8]
    files = glob.glob(data_dir + "precipitation/precipitation_20*")
    files.sort()   # sorting alphabetically is enough b/c nice naming scheme!

    # find the vegetation data closest to the requested date
    testf = data_dir + "precipitation/precipitation_%s" % (decidate)   # data for the day we'd really like
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

    precip_pct = []
    months = []
    n = 0
    for fname in fnames:
        found = False
        with open(fname, 'r') as fh:
            reader = csv.reader(fh, delimiter=" ")
            for row in reader:
                if (row[0] == lon and row[1] == lat):   # check for matching pixel
                    precip_pct.append(float(row[2]))
                    found = True
        if found:
            months.append(n)
        n+=1

    if len(precip_pct) < 10:
        print("no precipitation data avilible for this pixel") 
        return None

    # now fit a linear regression of the form y = mx+b
    x = np.array(months)
    y = np.array(precip_pct)
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
    N = 24
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

def toYearFraction(date):
    dt = datetime.datetime
    def sinceEpoch(date): # returns seconds since epoch
        return time.mktime(date.timetuple())
    s = sinceEpoch

    year = date.year
    startOfThisYear = dt(year=year, month=1, day=1)
    startOfNextYear = dt(year=year+1, month=1, day=1)

    yearElapsed = s(date) - s(startOfThisYear)
    yearDuration = s(startOfNextYear) - s(startOfThisYear)
    fraction = yearElapsed/yearDuration

    return date.year + fraction

def main():
    n_train = 600
    n_test = 10

    X, y = get_data(n_train+n_test)

    # Separate training and test sets
    X_train = X[0:-n_test]
    y_train = y[0:-n_test]
    X_test = X[-n_test:]
    y_test = y[-n_test:]

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
    load_all_data()
    X, y = get_data(n_train+n_test)

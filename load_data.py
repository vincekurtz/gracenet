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

    #print("===> Loading IRRIGATION data to memory")
    #irrigation_data = get_data_dict('irrigation/irrigation*', 'irr')
    #print("===> Loading POPULATION data to memory")
    #population_data = get_data_dict('population/population*', 'pop')

    print("===> Loading PRECIPITATION data to memory")
    precipitation_data = get_data_dict('precipitation/precipitation*', 'precip')

    print("===> Loading TEMPERATURE data to memory")
    temperature_data = get_data_dict('temperature/MOD11C3_LST*', 'temp')

    print("===> Loading VEGETATION data to memory")
    vegetation_data = get_data_dict('vegetation/MOD13C2_EVI_*', 'veg')

def get_regional_data():
    """
    Get training/testing data from plaintext files.
    Only use data from the conententla US (ish)
    Return X, y, where y is the GRACE anomoly and 
    X is the data we'll use to derive the anomoly.
    """
    X = []
    y = []

    print("===> Getting valid pixels")
    dates = valid_date_list()
    pixels = valid_pixel_list(dates)

    print("===> Generating dataset")

    for date in dates:
        anomolies = []
        precips = []  # store precipitation, temperature, and vegetation data
        temps = []    # we'll collapse these into 1d after we get all the pixels
        vegs = []
        for pixel in pixels:
            lat = pixel[1]
            lon = pixel[0]

            # grace anomoly --> output
            grace = grace_data[date][pixel]

            # other varialbes --> input
            precip = precipitation_data[date][pixel]
            temp = temperature_data[date][pixel]
            veg = vegetation_data[date][pixel]

            # Add to the datasets!
            anomolies.append(grace)
            precips.append(precip)
            precips.append(temp)
            vegs.append(veg)

        #print(len(anomolies))
        #print(len(precips+temps+vegs)/3)

    print(str(len(X)) + " datapoints")
    print("input dimensions: " + str(len(X[0])))

    return (X, y)

def valid_date_list():
    """
    Return a list of dates that have data for grace, precipitation,
    temperature, and vegetation.
    """
    dates = []
    for date in grace_data:
        if (date in precipitation_data and date in temperature_data and date in vegetation_data):
            dates.append(date)
    return dates

def valid_pixel_list(date_list):
    """
    Return a list of pixels in the contental US with precipitation, temperature, 
    vegetation, and grace data for all the given dates.
    """
    possiblepixels = set()
    badpixels = set()

    # get all grace pixels in the continental US
    for pixel in grace_data[(2002, 4)]:
        lon = pixel[0]
        lat = pixel[1]
        inbounds = True #(lat > 26 and lat < 49 and lon > -125 and lon < -67)
        if inbounds:
            possiblepixels.add(pixel)

    # now go back and filter out pixels that aren't in all the places
    for date in date_list:
        for pixel in possiblepixels:
            in_all_sets = (pixel in grace_data[date])
            if not in_all_sets:
                badpixels.add(pixel)

    valid_pixels = possiblepixels - badpixels  # pixels that are in possible but not in bad
    print(len(possiblepixels))
    print(len(badpixels))
    print(len(valid_pixels))
    return list(valid_pixels)

def get_data():
    """
    Get training/testing data from plaintext files.
    Return X, y, where y is the GRACE slope and 
    X is the data we'll use to derive the GRACE data.
    """
    X = []
    y = []

    max_n=200000

    print("===> Generating dataset")
    i = 0 # number of iterations
    for date in grace_data:
        for pixel in grace_data[(2004,1)]:   # use a consistent list of pixels
            lat = pixel[1]
            lon = pixel[0]
            # restrict to lower asia ish region
            try:
                # grace slope --> output
                grace = get_trend(pixel, date, grace_data)

                # other varialbes --> input
                # include both trend and average (over ~ 2 yrs)
                precip = get_trend(pixel, date, precipitation_data)
                temp = get_trend(pixel, date, temperature_data)
                veg = get_trend(pixel, date, vegetation_data)
                precipavg = get_average(pixel, date, precipitation_data)
                vegavg = get_average(pixel, date, vegetation_data)
                tempavg = get_average(pixel, date, temperature_data)

                if grace:  # it's useless to include data without an output!
                    # add to the master arrays of data
                    X.append([precip, precipavg, temp, tempavg, veg, vegavg])
                    y.append(grace)


            except KeyError:
                # sometimes we won't have enough corresponding data on some of the
                # extra variables. We'll just ignore that pixel/date pair in that case.
                pass
    

        n = len(X)
        print("Date %s / %s | Sample %s / %s " % (i, len(grace_data), n, max_n))

        if n > max_n:  # quit when we have enough samples
            break

        i+=1  # iteration counter

    print(str(len(X)) + " datapoints")
    print("input dimensions: " + str(len(X[0])))

    return (X, y)

def double_data(x_row, y_row):
    """
    Create and return an artificial dataset by adding
    gaussian noise to the given real data.
    """
    pass


def nearby_valid_date(desired_date, dictionary):
    """
    Sometimes we get a date (year, month, day) that does not exactly exist in
    another dictionary. We want to find a nearby date that does exist in that 
    dictionary, but is part of the same month.
    """
    for valid_date in dictionary:
        if (valid_date[0:2] ==  desired_date[0:2]):  # matching year and month
            return valid_date

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
            return (date.year, date.month)  # only use year and month since this is monthly data
    elif fformat == 'temp':
        # temperature
        def get_date(fname):
            regex = r'MOD11C3_LST_Day_CMG_([0-9]*)_([0-9]*)_monthly.csv'  # year, day of year format
            m = re.search(regex, fname)
            year = int(m.group(1))
            day_of_year = int(m.group(2))
            date = datetime.datetime(year, 1, 1) + datetime.timedelta(day_of_year-1)
            return (date.year, date.month)
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
            return (date.year, date.month)
    elif fformat == 'pop':
        # population
        def get_date(fname):
            regex = r'population_density_([0-9]*)_regridded.txt'
            m = re.search(regex, fname)
            year = int(m.group(1))
            return (year, 1)   # we only have population data on the year
    elif fformat == 'irr':
        # irritation
        def get_date(fname):
            regex = r'irrigation_pct_([0-9]*).csv'
            m = re.search(regex, fname)
            year = int(m.group(1))
            return (year, 1) 
    elif fformat == 'grace':
        # grace anomoly data
        def get_date(fname):
            regex = r'GRCTellus.JPL.([0-9]*).LND.RL05_1.DSTvSCS1411.txt'
            m = re.search(regex, fname)
            datestring = m.group(1)
            year = int(datestring[0:4])
            month = int(datestring[4:6])
            day = int(datestring[6:8])
            return (year, month) 

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

def get_trend(pixel, date, dataset):
    """
    Return a N month trend in the given dataset.
    """
    N = 24

    vals = []
    months = []
    n = 0
    bad_cnt = 0
    # generate lists of month numbers and values
    for i in range(N):
        try:
            vals.append(dataset[date][pixel])
            months.append(n)
        except KeyError:
            bad_cnt += 1  # ignore when we can't get a value
        n+=1
        date = previous_month(date)

    if bad_cnt > 15:
        return 0   # ingore if there are too few datapoints

    # now fit a linear regression
    x = np.array(months)
    y = np.array(vals)
    A = np.vstack([x, np.ones(len(x))]).T
    slope, y_int = np.linalg.lstsq(A, y)[0]

    return(slope)

def get_average(pixel, date, dataset):
    """
    Return an N month average in the given dataset.
    """
    N = 24

    vals = []
    bad_cnt = 0
    # generate lists of month numbers and values
    for i in range(N):
        try:
            vals.append(dataset[date][pixel])
        except KeyError:
            bad_cnt += 1  # ignore when we can't get a value
        date = previous_month(date)

    if bad_cnt > 15:
        return 0   # ingore if there are too few datapoints

    avg = np.average(vals)
    return avg

def previous_month(date):
    """
    Return the previous month for a given date
    """
    year = date[0]
    month = date[1]
    new_year = year
    new_month = month - 1

    if new_month == 0:
        new_year -= 1
        new_month = 12

    return (new_year, new_month)


def get_temperature_trend(pixel, date):
    """
    Return the 2 year tempearature trend for a given pixel
    and date. The trend should be over the N months before
    the given date.
    
    pixel should be a (lon, lat) touple. date should be a
    (year, month) touple
    """
    N = 24

    # get data for this pixel from these previous months
    lon = str(pixel[0])
    lat = str(pixel[1])

    this_temp = temperature_data[date][pixel]
    print(this_temp)

    return

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

def save_validation_data():
    """
    Save grace and input data in a csv file. 
    format: 
    LON LAT GRACESLOPE PRECIP TEMP VEG PRECIPAVG TEMPAVG VEGAVG
    """
    X = []  # input vars
    y = [] # grace

    with open('validation.csv', 'w') as fh: 
        writer = csv.writer(fh, delimiter=' ')
        writer.writerow(["HDR","long","lat","grace","precip","temp","veg","precipavg","tempavg","vegavg"])

        date = (2016,1)
        for pixel in grace_data[(2004,1)]:   # use a consistent list of pixels
            lat = pixel[1]
            lon = pixel[0]
            try:
                # grace slope --> output
                grace = get_trend(pixel, date, grace_data)

                # other varialbes --> input
                # include both trend and average (over ~ 2 yrs)
                precip = get_trend(pixel, date, precipitation_data)
                temp = get_trend(pixel, date, temperature_data)
                veg = get_trend(pixel, date, vegetation_data)
                precipavg = get_average(pixel, date, precipitation_data)
                vegavg = get_average(pixel, date, vegetation_data)
                tempavg = get_average(pixel, date, temperature_data)

                writer.writerow([lon, lat, grace, precip, temp, veg, precipavg, tempavg, vegavg])


            except KeyError:
                # sometimes we won't have enough corresponding data on some of the
                # extra variables. We'll just ignore that pixel/date pair in that case.
                pass

def main():
    X, y = get_data()

    # Separate training and test sets
    n_test = int(len(y)*0.10)   # 10% of the data for testing
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
    load_all_data()  # do this first since many functions reference global vars
    save_validation_data()


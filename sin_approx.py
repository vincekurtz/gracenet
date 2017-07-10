#!/usr/bin/env python

##
#
# Fit GRACE data to a sinusoidal model.
#
##

import numpy as np
from scipy.optimize import leastsq
import pylab as plt
from train import load_data
#from test import plot_stuff


def get_sinusoid_params(data):
    """
    For a given set of data points, fit a sinusoidal model
    with a slope (like A*sin(B*t+C) + D + E*t ) to the points.
    Return aplitude, phase, mean, and slope. The period is assumed
    to be 12 (12 months in a year).
    """
    N = 24 # number of data points
    m = np.linspace(0, N, N)  # monthwise grid
    
    guess_mean = np.mean(data)
    guess_amp = 3*np.std(data)/(2**0.5)
    guess_phase = 0
    guess_slope = 0
    w = 2*np.pi/12   # frequency is fixed for period of 12 months

    # starting point estimate
    data_first_guess = guess_amp*np.sin(w*m+guess_phase) + guess_mean + guess_slope*m

    # Define the function to optimize, in this case, we want to minimize the difference
    # between the actual data and our "guessed" parameters
    optimize_func = lambda x: x[0]*np.sin(2*np.pi/12*m+x[1]) + x[2] + x[3]*m - data
    est_amp, est_phase, est_mean, est_slope = leastsq(optimize_func, [guess_amp, guess_phase, guess_mean, guess_slope])[0]

    return((est_amp, est_phase, est_mean, est_slope))

def predict_next(data):
    """
    Use a sinusoidal model based on the data
    to predict the next point.
    """
    next_m = 24  # assume months 0-23 are in data
    est_amp, est_phase, est_mean, est_slope = get_sinusoid_params(data)  # fit the data

    # plug parameters into the model and get the desired point
    next_point = est_amp*np.sin(2*np.pi/12*next_m+est_phase) + est_mean + est_slope*next_m 

    return next_point


if __name__=="__main__":
    X, y = load_data('training_data.json')
    data = X[3]
    m = np.linspace(0, 22, 24)  # monthwise grid

    est_amp, est_phase, est_mean, est_slope = get_sinusoid_params(data)

    # recreate the fitted curve using the optimized parameters
    data_fit = est_amp*np.sin(2*np.pi/12*m+est_phase) + est_mean + est_slope*m

    plt.plot(data, '.', label='original data')
    plt.plot(data_fit, label='least square fit')
    plt.legend()
    plt.show()


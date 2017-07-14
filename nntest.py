#!/usr/bin/env python3

import numpy as np
from scipy.optimize import leastsq
from sklearn.neural_network import MLPRegressor
import pylab as plt

N = 1000 # number of data points
t = np.linspace(0, 4*np.pi, N)
data = 3.0*np.sin(t+0.001) + 0.5 + np.random.randn(N) # create artificial data with noise

net = MLPRegressor(
        solver='lbfgs',
        alpha=2,
        activation='logistic',
        hidden_layer_sizes=(5),
        random_state=1)

X = t.reshape(-1,1)  # input and output
y = data

net.fit(X, y)

pred = net.predict(X)
plt.plot(data, '.', color='0.75')
plt.plot(pred, 'r-', label='network output')
plt.legend()
plt.show()

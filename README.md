GRACE NET
=========

A neural network system for predicting groundwater
changes (like the TWS anomoly measured by the GRACE
sattelites) based on other easily measured parameters.

These parameters (for now) are vegetation, precipitation,
and temperature. 

### Dependencies
    - python3
    - python libraries
        - scikit-learn 
        - numpy
        - csv
        - random
        - glob
        - json
        - datetime
        - re (regex)
        - matplotlib (optional, for plotting learning curves)

### How the system works right now

`load_data.py` gathers data that is formated in space
separated csv files in the 'data' subfolders. It calculates
two year trends and two year averages for precipitation, 
temperature, vegetation, and GRACE anomoly. These are saved 
to two json files, one for training and one for testing. 

Test data is from the most recent few years, rather than a 
random sample, so we know the test set is separate from the
training set. 

`train.py` loads the training and test sets from the json 
files. It then performs basic network configuration, trains
on the training set, and performs basic tests. It then saves
parameters to disk as a 'pickel'. 

Optionally, gaussian noise can be added to the training data
in `train.py` to generate some artificial data. This isn't
necessary in the current configuration, where each pixel is
treated as a unique data point.

`test.py` contains tools for further testing of a trained
model. Right now it validates the model on data from 2014-2016,
saving the results as a csv file that can be fed into the
mapping system just like GRACE, CLM, or any of the other
data we have. 

There are many other functions that are not currently in use
throughout these files. This includes things like principle
component analysis (PCA) and scatterplot creation. While these
are not currently an essential part of the pipeline, they may
become useful again later.

#### Other files

`nntest.py` contains a simple example of fitting a sine-like
curve with a small network.

`sin_approx.py` fits grace trends (for ~2 years) to a sinusoid
model. This will (later) allow us to think about the slope only
for the network, but use the parameters (amplitude, phase,
period) of the sine model to approximate the actual fluxuations
in a given place at the end. 

### Future Work

A 500 neuron network with relu activation trained ~200000 data
points from approx. 2004-2006 performs decently when cross
validated against results from 2014-2016. Including more 
data points would be an obvious step towards improving performance,
as would using a larger network (probably with stronger
regularization). 

It should also be noted that this decent result came from a run
with a relatively poor (0.2 on training, 0.02 on testing) R^2 
score. So take the score with a grain of salt as far as accuracy
on the final map goes. Some work towards generating a better
measure of accuracy for this application wouldn't be a bad idea.

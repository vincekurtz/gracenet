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
two year trends for precipitation, temperature, vegetation,
and GRACE anomoly. These are saved to two json files, one for
training and one for testing. 

Test data is from the most recent few years, rather than a 
random sample, so we know the test set is separate from the
training set. 

`train.py` loads the training and test sets from the json 
files. It then performs basic network configuration, trains
on the training set, and performs basic tests. It then saves
parameters to disk as a 'pickel'. 

Optionally, gaussian noise can be added to the training data
in `train.py` to generate some artificial data.

`test.py` contains tools for further testing of a trained
model. It reads the model from a pickel and performs tests
from there. 

#### Other files

`nntest.py` contains a simple example of fitting a sine-like
curve with a small network.

`sin_approx.py` fits grace trends (for ~2 years) to a sinusoid
model. This will (later) allow us to think about the slope only
for the network, but use the parameters (amplitude, phase,
period) of the sine model to approximate the actual fluxuations
in a given place at the end. 

### Future Work

This is a work in progress. We need to rethink the network
archetecture, and probably gather more input variables
(evapotranspiration, irrigation, vegetation types, ...) 
before we are able to get good results. 

A 500x500 network with relu activation performs decently
on training data, but good performance on the test set
remains elusive. This implies some serious overfitting, 
but regularization kills the performance on the test set
without corresponding improvements on the training set. 
This suggests that we need more data sources in order to 
model groundwater changes at all acurately. 

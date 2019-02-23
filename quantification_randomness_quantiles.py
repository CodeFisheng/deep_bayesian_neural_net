import tensorflow as tf
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import numpy as np
import matplotlib as mp
import random as rd
import argparse
import os, sys
import csv
import math
import time
import matplotlib.pyplot as pl
from numpy import random, histogram2d, diff
from scipy.interpolate import interp2d
from scipy.stats import skew, kurtosis
#%matplotlib inline

feature = np.array(pd.read_csv('dresult/feature_all_10.csv', header = None))
uncertainty = np.array(pd.read_csv('dresult/uncertainty_all_10.csv', header = None))

#######  Temperatures
x = feature[:,45]
y = uncertainty[:,1]
nbins = 30
hist, bin_edges = np.histogram(x, nbins)

print hist
bin_edges
ylist = []
for i in range(nbins):
    ylist.append(y[(x >= bin_edges[i+0]) & (x < bin_edges[i+1])])
skewlist = []
ylist[0]
scipy.stats.skew(ylist[2])
for i in range(nbins):
    print ylist[i]
    skewlist.append(scipy.stats.skew(ylist[i]))
skewlist.shape

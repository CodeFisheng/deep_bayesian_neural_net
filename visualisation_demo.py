#################
#
#  this file is used to try alpha value or opacity as the
#  probability rather than color it self
#
#####################
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
import matplotlib.colors as colors
#%matplotlib inline

feature = np.array(pd.read_csv('dresult/feature_all_10.csv', header = None))
uncertainty = np.array(pd.read_csv('dresult/uncertainty_all_10.csv', header = None))
#rowlimit = feature.shape[0]/5
#feature = feature[0:rowlimit]
#uncertainty = uncertainty[0:rowlimit]
###################### temperature ############################


hoursamples = []
hoursamples.append(np.array(uncertainty[feature[:,22] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,23] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,24] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,25] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,26] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,27] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,28] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,29] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,30] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,31] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,32] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,33] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,34] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,35] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,36] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,37] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,38] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,39] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,40] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,41] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,42] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,43] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,44] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,21] == 1, 1]))
hoursamples = np.array(hoursamples)
hsamples = []
dsamples = []
for i in range(hoursamples.shape[0]):
    for j in range(hoursamples[i].shape[0]):
        hsamples.append(i)
        dsamples.append(hoursamples[i][j])
hsamples = np.array(hsamples)
dsamples = np.array(dsamples)
nbins = [24, 50]
H, xedges, yedges = histogram2d(hsamples, dsamples, bins=nbins, normed = True)
HH = H.T
for col in range(nbins[0]):
    coe = np.sum(HH[:,col])
    if col == nbins[0]:
        coe = np.sum(HH[:,0])
    if coe == 0:
        continue;
    HH[:,col] = HH[:,col]/coe
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2
xedges
xedges = list(xedges)
xedges[1]
xedges.append(23.96)
xedges = np.array(xedges)
xedges
xcenters = centers(xedges)
ycenters = centers(yedges)
H.shape
HH.shape
HH[:,0].shape
HHH = np.concatenate([HH, HH[:,0].reshape(50,1)], axis = 1)
HHH.shape
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
pl.clf()
pl.colorbar(pl.contourf(xcenters, ycenters, HHH, 100, cmap=pl.cm.hot))
pd.DataFrame(xcenters).to_csv('vresult/hour_x.csv', header = None)
pd.DataFrame(ycenters).to_csv('vresult/hour_y.csv', header = None)
pd.DataFrame(H).to_csv('vresult/hour_z.csv', header = None)
#pl.show()
pl.savefig('imgresult/hour.png')

def my_cmap():
    # 白青绿黄红
    cdict = ['#FF0000', '#FF1100', '#FF2200', '#FF3300', '#FF4400', '#FF5500', \
    '#FF6600', '#FF7700', '#FF8800', '#FF9900', '#FFAA00', '#FFBB00', '#FFCC00', \
    '#FFDD00', '#FFEE00', '#FFFF00', '#EEFF00', '#DDFF00', '#CCFF00', '#BBFF00', \
    '#AAFF00', '#99FF00', '#88FF00', '#77FF00', '#66FF00', '#55FF00', '#44FF00', \
    '#33FF00', '#22FF00', '#11FF00', '#00FF00']
    cdict.reverse()
    # 按照上面定义的colordict，将数据分成对应的部分，indexed：代表顺序
    return colors.ListedColormap(cdict, 'indexed')
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
#pl.clf()
#pl.colorbar(pl.contourf(xcenters, ycenters, HH, 15, alpha = [0.3, 1], cmap=my_cmap()))#pl.cm.binary))
pl.show()

help(pl.colorbar)

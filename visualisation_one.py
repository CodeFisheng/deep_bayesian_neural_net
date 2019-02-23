############################# this code unfinished #######################
# seperately apply marginal interal sampling on different households


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
#%matplotlib inline

feature = np.array(pd.read_csv('dresult/feature_all.csv', header = None))
uncertainty = np.array(pd.read_csv('dresult/uncertainty_all.csv', header = None))

###################### temperature ############################
x = feature[:,45]
y = uncertainty[:,1]
nbins = 30
H, xedges, yedges = histogram2d(x, y, bins=nbins, normed = True)
for col in range(nbins):
    coe = np.sum(H[:,col])
    H[:,col] = H[:,col]/coe
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2
xcenters = centers(xedges)
ycenters = centers(yedges)
pdf = interp2d(xcenters, ycenters, H)
H.shape
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
pl.clf()
pl.colorbar(pl.contourf(xcenters[0:26], ycenters, H[:, 0:26], 300, cmap=pl.cm.hot))
pd.DataFrame(xcenters).to_csv('vresult/xmatrix_all.csv', header = None)
pd.DataFrame(ycenters).to_csv('vresult/ymatrix_all.csv', header = None)
pd.DataFrame(H).to_csv('vresult/zmatrix_all.csv', header = None)
pl.show()
pl.savefig('heat_all.png')
# from mpl_toolkits.mplot3d import axes3d
# fig = pl.figure()
# ax = fig.gca(projection='3d')
# I = np.ones((30,30))
# xx = I*xcenters
# yy = (I*ycenters).transpose([1,0])
# zz = H
# cset = ax.plot_surface(xx,yy,zz, cmap=pl.cm.coolwarm)
#
# ax.clabel(cset, fontsize=9, inline=10)

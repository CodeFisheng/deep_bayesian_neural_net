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

feature = np.array(pd.read_csv('dresult/feature_all_10.csv', header = None))
uncertainty = np.array(pd.read_csv('dresult/uncertainty_all_10.csv', header = None))
#rowlimit = feature.shape[0]/5
#feature = feature[0:rowlimit]
#uncertainty = uncertainty[0:rowlimit]
###################### temperature ############################
x = feature[:,45]
y = uncertainty[:,1]
nbins = [30,30]
H, xedges, yedges = histogram2d(x, y, bins=nbins, normed = True)
HH = H.T
for col in range(nbins[0]):
    coe = np.sum(HH[:,col])
    if coe == 0:
        continue;
    HH[:,col] = HH[:,col]/coe
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2
xcenters = centers(xedges)
ycenters = centers(yedges)
H.shape
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
pl.clf()
pl.colorbar(pl.contourf(xcenters, ycenters, HH, 100, cmap=pl.cm.hot))
pd.DataFrame(xcenters).to_csv('vresult/temperature_x.csv', header = None)
pd.DataFrame(ycenters).to_csv('vresult/temperature_y.csv', header = None)
pd.DataFrame(H).to_csv('vresult/temperature_z.csv', header = None)
pl.show()
pl.savefig('imgresult/temperature.png')

###################### humidity ############################
x = feature[:,46]
y = uncertainty[:,1]
nbins = [30, 30]
H, xedges, yedges = histogram2d(x, y, bins=nbins, normed = True)
HH = H.T
for col in range(nbins[0]):
    coe = np.sum(HH[:,col])
    if coe == 0:
        continue;
    HH[:,col] = HH[:,col]/coe
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2
xcenters = centers(xedges)
ycenters = centers(yedges)
H.shape
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
pl.clf()
pl.colorbar(pl.contourf(xcenters, ycenters, HH, 100, cmap=pl.cm.hot))
pd.DataFrame(xcenters).to_csv('vresult/humidity_x.csv', header = None)
pd.DataFrame(ycenters).to_csv('vresult/humidity_y.csv', header = None)
pd.DataFrame(H).to_csv('vresult/humidity_z.csv', header = None)
#pl.show()
pl.savefig('imgresult/humidity.png')


###################### WindSpeed ############################
x = feature[:,47]
y = uncertainty[:,1]
nbins = [30,30]
H, xedges, yedges = histogram2d(x, y, bins=nbins, normed = True)
HH = H.T
for col in range(nbins[0]):
    coe = np.sum(HH[:,col])
    if coe == 0:
        continue;
    HH[:,col] = HH[:,col]/coe
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2
xcenters = centers(xedges)
ycenters = centers(yedges)
H.shape
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
pl.clf()
pl.colorbar(pl.contourf(xcenters, ycenters, HH, 100, cmap=pl.cm.hot))
pd.DataFrame(xcenters).to_csv('vresult/windspeed_x.csv', header = None)
pd.DataFrame(ycenters).to_csv('vresult/windspeed_y.csv', header = None)
pd.DataFrame(H).to_csv('vresult/windspeed_z.csv', header = None)
#pl.show()
pl.savefig('imgresult/windspeed.png')


###################### rainfall ############################
x = feature[:,48]
y = uncertainty[:,1]
nbins = [30,30]
H, xedges, yedges = histogram2d(x, y, bins=nbins, normed = True)
HH = H.T
for col in range(nbins[0]):
    coe = np.sum(HH[:,col])
    if coe == 0:
        continue;
    HH[:,col] = HH[:,col]/coe
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2
xcenters = centers(xedges)
ycenters = centers(yedges)
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
pl.clf()
pl.colorbar(pl.contourf(xcenters, ycenters, HH, 100, cmap=pl.cm.hot))
pd.DataFrame(xcenters).to_csv('vresult/rainfall_x.csv', header = None)
pd.DataFrame(ycenters).to_csv('vresult/rainfall_y.csv', header = None)
pd.DataFrame(H).to_csv('vresult/rainfall_z.csv', header = None)
#pl.show()
pl.savefig('imgresult/rainfall.png')


###################### dayofweek ############################
monday = uncertainty[feature[:,1] == 1, 1]
tuesday = uncertainty[feature[:,2] == 1, 1]
wednesday = uncertainty[feature[:,3] == 1, 1]
thursday = uncertainty[feature[:,4] == 1, 1]
friday = uncertainty[feature[:,5] == 1, 1]
saturday = uncertainty[feature[:,6] == 1, 1]
sunday = uncertainty[feature[:,7] == 1, 1]
nbins = 200

pl.clf()
yd1, xd1, patches = pl.hist(monday, nbins, normed=1, facecolor='red', alpha=0.1)
xd1 = xd1[0:-1] + 0.5*np.diff(xd1)
ytotal = np.sum(yd1)
yd1 = 100*yd1/ytotal

yd2, xd2, patches = pl.hist(tuesday, nbins, normed=1, facecolor='red', alpha=0.1)
xd2 = xd2[0:-1] + 0.5*np.diff(xd2)
ytotal = np.sum(yd2)
yd2 = 100*yd2/ytotal

yd3, xd3, patches = pl.hist(wednesday, nbins, normed=1, facecolor='red', alpha=0.1)
xd3 = xd3[0:-1] + 0.5*np.diff(xd3)
ytotal = np.sum(yd3)
yd3 = 100*yd3/ytotal

yd4, xd4, patches = pl.hist(thursday, nbins, normed=1, facecolor='red', alpha=0.1)
xd4 = xd4[0:-1] + 0.5*np.diff(xd4)
ytotal = np.sum(yd4)
yd4 = 100*yd4/ytotal

yd5, xd5, patches = pl.hist(friday, nbins, normed=1, facecolor='red', alpha=0.1)
xd5 = xd5[0:-1] + 0.5*np.diff(xd5)
ytotal = np.sum(yd5)
yd5 = 100*yd5/ytotal

yd6, xd6, patches = pl.hist(saturday, nbins, normed=1, facecolor='red', alpha=0.1)
xd6 = xd6[0:-1] + 0.5*np.diff(xd6)
ytotal = np.sum(yd6)
yd6 = 100*yd6/ytotal

yd7, xd7, patches = pl.hist(sunday, nbins, normed=1, facecolor='red', alpha=0.1)
xd7 = xd7[0:-1] + 0.5*np.diff(xd7)
ytotal = np.sum(yd7)
yd7 = 100*yd7/ytotal

pl.clf()
pl.plot(xd1, yd1, 'r')
pl.plot(xd2, yd2, 'orange')
pl.plot(xd3, yd3, 'yellow')
pl.plot(xd4, yd4, 'g')
pl.plot(xd5, yd5, 'blue')
pl.plot(xd6, yd6, 'purple')
pl.plot(xd7, yd7, 'black')

#pl.show()
pl.savefig('imgresult/dayofweek.png')


###################### months ############################
Jan = uncertainty[feature[:,9] == 1, 1]
Feb = uncertainty[feature[:,10] == 1, 1]
Mar = uncertainty[feature[:,11] == 1, 1]
Apr = uncertainty[feature[:,12] == 1, 1]
May = uncertainty[feature[:,13] == 1, 1]
June = uncertainty[feature[:,14] == 1, 1]
July = uncertainty[feature[:,15] == 1, 1]
Aug = uncertainty[feature[:,16] == 1, 1]
Sep = uncertainty[feature[:,17] == 1, 1]
Octo = uncertainty[feature[:,18] == 1, 1]
Nov = uncertainty[feature[:,19] == 1, 1]
Dec = uncertainty[feature[:,20] == 1, 1]
nbins = 200

ym1, xm1, patches = pl.hist(Jan, nbins, normed=1, facecolor='red', alpha=0.1)
xm1 = xm1[0:-1] + 0.5*np.diff(xm1)
ytotal = np.sum(ym1)
ym1 = 100*ym1/ytotal
pl.plot(xm1, ym1, 'r', label = 'Jan')

ym2, xm2, patches = pl.hist(Feb, nbins, normed=1, facecolor='red', alpha=0.1)
xm2 = xm2[0:-1] + 0.5*np.diff(xm2)
ytotal = np.sum(ym2)
ym2 = 100*ym2/ytotal
pl.plot(xm2, ym2, 'r', label = 'Feb')

ym3, xm3, patches = pl.hist(Mar, nbins, normed=1, facecolor='red', alpha=0.1)
xm3 = xm3[0:-1] + 0.5*np.diff(xm3)
ytotal = np.sum(ym3)
ym3 = 100*ym3/ytotal
pl.plot(xm3, ym3, 'r', label = 'Mar')

ym4, xm4, patches = pl.hist(Apr, nbins, normed=1, facecolor='red', alpha=0.1)
xm4 = xm4[0:-1] + 0.5*np.diff(xm4)
ytotal = np.sum(ym4)
ym4 = 100*ym4/ytotal
pl.plot(xm4, ym4, 'r', label = 'Apr')

ym5, xm5, patches = pl.hist(May, nbins, normed=1, facecolor='red', alpha=0.1)
xm5 = xm5[0:-1] + 0.5*np.diff(xm5)
ytotal = np.sum(ym5)
ym5 = 100*ym5/ytotal
pl.plot(xm5, ym5, 'r', label = 'May')

ym6, xm6, patches = pl.hist(June, nbins, normed=1, facecolor='red', alpha=0.1)
xm6 = xm6[0:-1] + 0.5*np.diff(xm6)
ytotal = np.sum(ym6)
ym6 = 100*ym6/ytotal
pl.plot(xm6, ym6, 'r', label = 'June')

ym7, xm7, patches = pl.hist(July, nbins, normed=1, facecolor='red', alpha=0.1)
xm7 = xm7[0:-1] + 0.5*np.diff(xm7)
ytotal = np.sum(ym7)
ym7 = 100*ym7/ytotal
pl.plot(xm7, ym7, 'r', label = 'July')

ym8, xm8, patches = pl.hist(Aug, nbins, normed=1, facecolor='red', alpha=0.1)
xm8 = xm8[0:-1] + 0.5*np.diff(xm8)
ytotal = np.sum(ym8)
ym8 = 100*ym8/ytotal
pl.plot(xm8, ym8, 'r', label = 'Aug')

ym9, xm9, patches = pl.hist(Sep, nbins, normed=1, facecolor='red', alpha=0.1)
xm9 = xm9[0:-1] + 0.5*np.diff(xm9)
ytotal = np.sum(ym9)
ym9 = 100*ym9/ytotal
pl.plot(xm9, ym9, 'r', label = 'Sep')

ym10, xm10, patches = pl.hist(Octo, nbins, normed=1, facecolor='red', alpha=0.1)
xm10 = xm10[0:-1] + 0.5*np.diff(xm10)
ytotal = np.sum(ym10)
ym10 = 100*ym10/ytotal
pl.plot(xm10, ym10, 'r', label = 'Octo')

ym11, xm11, patches = pl.hist(Nov, nbins, normed=1, facecolor='red', alpha=0.1)
xm11 = xm11[0:-1] + 0.5*np.diff(xm11)
ytotal = np.sum(ym11)
ym11 = 100*ym11/ytotal
pl.plot(xm11, ym11, 'r', label = 'Nov')

ym12, xm12, patches = pl.hist(Dec, nbins, normed=1, facecolor='red', alpha=0.1)
xm12 = xm12[0:-1] + 0.5*np.diff(xm12)
ytotal = np.sum(ym12)
ym12 = 100*ym12/ytotal
pl.plot(xm12, ym12, 'black', label = 'Dec')
#pl.show()

pl.clf()
pl.plot(xm1, ym1, 'r')
pl.plot(xm2, ym2, 'orange')
pl.plot(xm3, ym3, 'yellow')
pl.plot(xm4, ym4, 'g')
pl.plot(xm5, ym5, 'blue')
pl.plot(xm6, ym6, 'purple')
pl.plot(xm7, ym7, 'black')
pl.plot(xm8, ym8, 'yellow')
pl.plot(xm9, ym9, 'g')
pl.plot(xm10, ym10, 'blue')
pl.plot(xm11, ym11, 'purple')
pl.plot(xm12, ym12, 'black')


pl.savefig('imgresult/month.png')

###################### hours ############################
hoursamples = []
hoursamples.append(np.array(uncertainty[feature[:,21] == 1, 1]))
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
hoursamples = np.array(hoursamples)
hsamples = []
dsamples = []
for i in range(hoursamples.shape[0]):
    for j in range(hoursamples[i].shape[0]):
        hsamples.append(i)
        dsamples.append(hoursamples[i][j])
hsamples = np.array(hsamples)
dsamples = np.array(dsamples)
nbins = [25, 30]
H, xedges, yedges = histogram2d(hsamples, dsamples, bins=nbins, normed = True)
HH = H.T
for col in range(nbins[0]):
    coe = np.sum(HH[:,col])
    if coe == 0:
        continue;
    HH[:,col] = HH[:,col]/coe
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2
xcenters = centers(xedges)
ycenters = centers(yedges)
H.shape
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
pl.clf()
pl.colorbar(pl.contourf(xcenters, ycenters, HH, 100, cmap=pl.cm.hot))
pd.DataFrame(xcenters).to_csv('vresult/hour_x.csv', header = None)
pd.DataFrame(ycenters).to_csv('vresult/hour_y.csv', header = None)
pd.DataFrame(H).to_csv('vresult/hour_z.csv', header = None)
#pl.show()
pl.savefig('imgresult/hour.png')


###################### months-contour ############################
hoursamples = []
hoursamples.append(np.array(uncertainty[feature[:,9] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,10] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,11] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,12] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,13] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,14] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,15] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,16] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,17] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,18] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,19] == 1, 1]))
hoursamples.append(np.array(uncertainty[feature[:,20] == 1, 1]))
hoursamples = np.array(hoursamples)
hsamples = []
dsamples = []
for i in range(hoursamples.shape[0]):
    for j in range(hoursamples[i].shape[0]):
        hsamples.append(i)
        dsamples.append(hoursamples[i][j])
hsamples = np.array(hsamples)
dsamples = np.array(dsamples)
nbins = [12, 30]
H, xedges, yedges = histogram2d(hsamples, dsamples, bins=nbins, normed = True)
HH = H.T
for col in range(nbins[0]):
    coe = np.sum(HH[:,col])
    if coe == 0:
        continue;
    HH[:,col] = HH[:,col]/coe
def centers(edges):
    return edges[:-1] + diff(edges[:2])/2
xcenters = centers(xedges)
ycenters = centers(yedges)
H.shape
# x = np.array(pd.read_csv('xmatrix.csv', header = None))
# y = np.array(pd.read_csv('ymatrix.csv', header = None))
# z = np.array(pd.read_csv('zmatrix.csv', header = None))
pl.clf()
H.shape
pl.colorbar(pl.contourf(np.arange(1,13), ycenters, HH, 100, cmap=pl.cm.hot))
pd.DataFrame(xcenters).to_csv('vresult/monthcontour_x.csv', header = None)
pd.DataFrame(ycenters).to_csv('vresult/monthcontour_y.csv', header = None)
pd.DataFrame(H).to_csv('vresult/monthcontour_z.csv', header = None)
#pl.show()
pl.savefig('imgresult/monthcontour.png')


###################### holiday ############################
holiday = np.array(uncertainty[feature[:,8] == 1, 1])
workday = np.array(uncertainty[feature[:,8] == 0, 1])
np.min(holiday)
np.min(workday)
nbins = 100
pl.clf()
np.mean(holiday)
np.mean(workday)
holiN,holiB,c = pl.hist(holiday, bins = nbins, range = [0, 0.5], normed = True)
workN,workB,c = pl.hist(0-workday, bins = nbins, range = [-0.5, 0], normed = True)
impact = np.convolve(holiN, workN, mode = 'full')
impact.shape
pl.plot(range(0,199), impact)
#pl.show()
pl.savefig('imgresult/holidayimpact.png')


###################################################parameter estimate: define prior distrib ution, not data-driven

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

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
# class County:
#     def __init__(self,parsename):
#         self.parsename = parsename
#         dataframe = xls.parse(parsename)
#         self.data = dataframe
#     def disp_all(self):
#         print self.dataframe
#     def get_all(self):
#         return self.dataframe
# xls = pd.ExcelFile('No-drybulb-dewpoint-short-dataset-CT.xlsx',header = None)
# Zonal = []
# Zonal.append(County('CT'))
# ZonalNum = 1

print "setting parameters ............."
num_epochs = 12000 # training epoches for each customer samples
num_realisations = 5000
out_thresh = num_epochs - 100
day_steps_f = 24
#val_rate_f = 0.15
test_batch_size_f = 0*day_steps_f # days of a batch
valid_batch_size_f = 28*day_steps_f
train_batch_size_f = 28*day_steps_f
n_output_f = 1
n_hidden_f_1 = 100
n_hidden_f_2 = 100
n_hidden_f_3 = 100
n_hidden_f_4 = 100
tao_f = 0.1
gap_test_f = 10
batch_size_f = test_batch_size_f # in this version, batch_size set same
preserve_f = 0#16114 ## amount of first time points without complete features
_dropout_train = 0.5
_dropout_test = 1.0
ZonalNum = 1
# DEMAND MATRIX 9 X LENGTH, 9: INC is total, index with 0, other substations are from 1 -> 8



############################||||||||||||||||||||||||||data loading

print "loading data ..........."
#ISO_name = 'No-drybulb-dewpoint-short-dataset-CT.csv'
#HOME_name = 'data/MAC005540.csv'
count = 0
for root, dirs, filenames in os.walk('./data/'):
    for fname in filenames:
        dbslice = pd.read_csv('./data/' + fname)
        if count == 0:
            xls = dbslice
        else:
            xls = pd.concat([xls,dbslice], axis = 0)
        count = count + 1
        if count > 10:
            print 'reach max file number'
            break;
xls.shape
rows_f = xls.shape[0]
columns_f = xls.shape[1]
database_f = np.array(xls)
for i in range(rows_f):
    for j in range(columns_f):
        database_f[i,j] = np.float(database_f[i,j])
totalen_f = rows_f
#print database_f[:,0]
n_input_f = columns_f - 1
data_norm = np.max(database_f, axis = 0)
database_f = database_f/data_norm
#print totalen_f
db_f = database_f
#print db_f
#define id arrays
test_id_f = np.array(test_batch_size_f)
valid_id_f = np.array(2*valid_batch_size_f)
train_id_f = np.array(totalen_f - test_batch_size_f - valid_batch_size_f)

#give values to id arrays
rang = range(preserve_f, totalen_f - test_batch_size_f)
valid_id_f = rd.sample(rang,2*valid_batch_size_f)
test_id_f = np.array(range(totalen_f - test_batch_size_f,totalen_f))
train_id_f = set(range(preserve_f, totalen_f - test_batch_size_f)) - set(valid_id_f)

#sort three id array
valid_id_f = np.sort(valid_id_f)
test_id_f = np.sort(test_id_f)
train_id_f = np.array(list(train_id_f))
def train_data_gen():
    X = np.zeros((train_batch_size_f,ZonalNum,n_input_f))
    Y = np.zeros((train_batch_size_f,ZonalNum,n_output_f))
    count = 0
    rang = range(0,train_id_f.shape[0])
    train_rd = rd.sample(rang,train_batch_size_f)
    train_rd = np.sort(train_rd)
    for i in train_rd:
        j = train_id_f[i]
        Y[count] = db_f[j,:1]
        X[count] = db_f[j,1:]
        count = count + 1
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    return (X,Y)

def valid_data_gen():
    X = np.zeros((train_batch_size_f,ZonalNum,n_input_f))
    Y = np.zeros((train_batch_size_f,ZonalNum,n_output_f))
    count = 0
    rang = range(0,valid_id_f.shape[0])
    valid_rd = rd.sample(rang,train_batch_size_f)
    valid_rd = np.sort(valid_rd)
    for i in valid_rd:
        j = valid_id_f[i]
        Y[count] = db_f[j,:1]
        X[count] = db_f[j,1:]
        count = count + 1
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    return (X,Y)

def test_data_gen():
    X = np.zeros((test_batch_size_f,ZonalNum,n_input_f))
    Y = np.zeros((test_batch_size_f,ZonalNum,n_output_f))
    count = 0
    for i in test_id_f:
        Y[count] = db_f[i,:1]
        X[count] = db_f[i,1:]
        count = count + 1
    X = X.astype(np.float32)
    Y = Y.astype(np.float32)
    return (X,Y)

print 'Construct Neural Nets'
_X_f = tf.placeholder(tf.float32, [None, ZonalNum, n_input_f])
_Y_f = tf.placeholder(tf.float32, [None, ZonalNum, n_output_f])
_Dropout_f = tf.placeholder(tf.float32)


# Create model
def MLP(x, _dropout, weights, biases):

    x = tf.reshape(x, [-1, n_input_f])

    # Hidden layer with RELU activation
    x = tf.nn.dropout(x, _dropout)
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)
    layer_1 = tf.nn.dropout(layer_1,_dropout)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)
    layer_2 = tf.nn.dropout(layer_2,_dropout)

    # Hidden layer with RELU activation
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)
    layer_3 = tf.nn.dropout(layer_3,_dropout)

    # Hidden layer with RELU activation
    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.sigmoid(layer_4)
    # Output layer with linear activation
    result = tf.matmul(layer_3, weights['out']) + biases['out']
    result = tf.nn.sigmoid(result)
    return result
# MLP
weights_f = {
    'h1': tf.Variable(tf.random_normal([n_input_f, n_hidden_f_1]), name = "wf1"),
    'h2': tf.Variable(tf.random_normal([n_hidden_f_1, n_hidden_f_2]), name = "w_f_2"),
    'h3': tf.Variable(tf.random_normal([n_hidden_f_2, n_hidden_f_3]), name = "w_f_3"),
    'h4': tf.Variable(tf.random_normal([n_hidden_f_3, n_hidden_f_4]), name = "w_f_4"),
    'out': tf.Variable(tf.random_normal([n_hidden_f_4, n_output_f]), name = "w_o")
}
biases_f = {
    'b1': tf.Variable(tf.random_normal([n_hidden_f_1]), name = "b_f_1"),
    'b2': tf.Variable(tf.random_normal([n_hidden_f_2]), name = "b_f_2"),
    'b3': tf.Variable(tf.random_normal([n_hidden_f_3]), name = "b_f_3"),
    'b4': tf.Variable(tf.random_normal([n_hidden_f_4]), name = "b_f_4"),
    'out': tf.Variable(tf.random_normal([n_output_f]), name = "b_o")
}
pred_f = MLP(_X_f, _Dropout_f, weights_f, biases_f)
reshaped_results_f = tf.reshape(_Y_f, [-1])
reshaped_outputs_f = tf.reshape(pred_f, [-1])
#coef = 0.0001
#closs = coef*tf.nn.l2_loss(weights['h1']) + coef*tf.nn.l2_loss(weights['h2']) + coef*tf.nn.l2_loss(weights['h3']) + coef*tf.nn.l2_loss(weights['out'])
cost_f = tf.reduce_mean(tf.pow(reshaped_results_f - reshaped_outputs_f,2))
#cost = tf.nn.l2_loss(reshaped_results-reshaped_outputs)
optimizer_f = tf.train.AdamOptimizer(learning_rate=0.005, beta1 = 0.8, beta2 = 0.7).minimize(cost_f)

def maxe(predictions, targets):
    return max(abs(predictions-targets))

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def mape(predictions, targets):
    return np.mean(abs(predictions-targets)/targets)

outlist = []
realist = []
samples = []
samples_res = []
features = []
uncertainties = []
kind = 0
time1 = time.time()
# generate test data
test_x,test_y = test_data_gen()
# Initializing the variables
init = tf.initialize_all_variables()
reg = 0                        #,"wf2":w_f_2,"wf3":w_f_3,"wf4":w_f_4,"wo":w_o,"bf1":b_f_1,"bf2":b_f_2,"bf3":b_f_3,"bf4":b_f_4,"bo":b_o})
print "Start"
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    # Create a summary to monitor cost function
    #tf.scalar_summary("loss", cost_f)
    # Merge all summaries to a single operator
    #merged_summary_op = tf.merge_all_summaries()

    # tensorboard info.# Set logs writer into folder /tmp/tensorflow_logs
    #summary_writer = tf.train.SummaryWriter('path/to/logs', graph_def=sess.graph_def)

    #initialize all variables in the model
    sess.run(init)
    saver = tf.train.Saver({"wf1":weights_f['h1'],"wf2":weights_f['h2'],"wf3":weights_f['h3'],"wf4":weights_f['h4'],"wfo":weights_f['out'],"bf1":biases_f['b1'],"bf2":biases_f['b2'],"bf3":biases_f['b3'],"bf4":biases_f['b4'],"bfo":biases_f['out']})
    # for k in range(num_epochs):
    #     #print traindays
    #
    #     #print 'Training'
    #     if rd.random() < 2*valid_batch_size_f/totalen_f:
    #         X,Y = valid_data_gen()
    #         cs, _ = sess.run([cost_f,optimizer_f], feed_dict = {_X_f:X, _Y_f:Y, _Dropout_f:_dropout_train})
    #     else:
    #         X,Y = train_data_gen()
    #         cs, _ = sess.run([cost_f,optimizer_f],feed_dict={_X_f:X,_Y_f:Y,_Dropout_f: _dropout_train})
    #         #summary2 = sess.run([cost_f,optimizer_f, merged_summary_op],feed_dict={_X_f:X,_Y_f:Y,_Dropout_f: _dropout_train})
    #     if k % 1000 == 0:
    #         print "Iter " + str(k) + " ---- Process: " + "{:.2f}".format(100*float(k)/float(num_epochs)) + "%, loss = "+"{:.4f}".format(100*np.sqrt(cs))+"%"
    #         #summary_writer.add_summary(summary, k)
    #         #summary_writer.add_summary
    #     # if (k >= out_thresh) & (k % gap_test_f == 0):
    #     #     #print test_x
    #     #     err, reshaped_pred, reshaped_res = sess.run([cost_f, reshaped_outputs_f, reshaped_results_f],feed_dict = {_X_f:test_x,_Y_f:test_y,_Dropout_f: _dropout_test} )
    #     #     print "RMSE = " + str(np.sqrt(err))
    #     #     outlist.append(reshaped_pred)
    #     #     realist.append(reshaped_res)
    #     #     kind = kind + 1
    # saver.save(sess, "models/HOME_model_all.ckpt")#%HOME_name[10:14])
    saver.restore(sess, "models/HOME_model_all_10.ckpt")#%HOME_name[10:14])
    # saver.save(sess, "ISO_model_%d.ckpt"%num_epochs)
    # saver.restore(sess, "ISO_model_%d.ckpt"%num_epochs)
    print 'Testing'
    ######################################### resulting for load forecasting
    for k in range(num_realisations):
        reshaped_pred, reshaped_res = sess.run([reshaped_outputs_f, reshaped_results_f], feed_dict = {_X_f:test_x,_Y_f:test_y,_Dropout_f: _dropout_train})
        samples.append(reshaped_pred)
        samples_res.append(reshaped_res)

outlist = np.array(outlist)
realist = np.array(realist)
samples = np.array(samples)
samples_res = np.array(samples_res)
RList = []
rmseList = []
maxeList = []
mapeList = []
outlist = outlist * (np.max(database_f[:,:,0]))
realist = realist * (np.max(database_f[:,:,0]))
samples = samples * (np.max(database_f[:,:,0]))
samples_res = samples_res * (np.max(database_f[:,:,0]))
for i in range(kind):
	out = np.array(outlist[i])
	res = np.array(realist[i])
	RList.append(np.corrcoef(out,res)[0,1])
	rmseList.append(rmse(out,res))
	maxeList.append(maxe(out,res))
	mapeList.append(mape(out,res))
time2 = time.time()
#print "time(s): "+str(time2-time1)
#print "rmse = " + str(np.mean(rmseList))
#print samples
samples = np.sort(samples, axis = 0)
Plist = []
for i in range(0,9):
	ind = 10*(i+1)
	Plist.append(np.percentile(samples, ind, axis = 0))
def pinball_loss(A,B,tao):
	cost = 0.0
	A = A.reshape([-1])
	B = B.reshape([-1])
	for i in range(A.shape[0]):
	    if A[i]-B[i]>=0:
	        tmp = (A[i]-B[i])*(tao)
	    else:
	        tmp = (B[i]-A[i])*(1.0-tao)
	    cost = tmp+cost
	#print cost
	#print ncost_s3
	return cost
taolist = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
nloss_pb_total = 0.0
loss_pb_total = 0.0
for t in xrange(0,9):
	tao = taolist[t]
	loss_pb = pinball_loss(samples_res[0], Plist[t], tao)
	loss_pb_total = loss_pb_total + loss_pb
print "dropout = %.4f, loss_pb_total = %.4f"%(_dropout_train, loss_pb_total)
pinballist.append(loss_pb_total)
#DataFrame(pinballist).to_csv("PINBALL-onelayer.csv")

print "loss_pb_total = ",loss_pb_total
print 'Visualization'
N = 1
x = np.linspace(0, 672, 672)
oneday_x = x[0:24]
vals = [1,2,3,4] # Values to iterate over and add/subtract from y.
pl.rc('font', family='serif')
fig = pl.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('The x values')
ax.set_ylabel('The y values')
#ax.set_ylim([0,1.8])
pl.rc('font', family = 'serif', serif = 'Times')
pl.rc('xtick', labelsize = 8)
pl.rc('ytick', labelsize = 8)
pl.rc('axes', labelsize = 8)
###################################### one month case
for i, val in enumerate(vals):
    alpha = 0.5*(i+1)/len(vals) # Modify the alpha value for each iteration.
    if i == 0:
        ax.fill_between(x, Plist[8], Plist[0], color='red', alpha=alpha)
    if i == 1:
        ax.fill_between(x, Plist[7], Plist[1], color='red', alpha=alpha/2)
    elif i == 2:
        ax.fill_between(x, Plist[6], Plist[2], color='red', alpha=alpha/2)
    else:
        ax.fill_between(x, Plist[5], Plist[3], color='red', alpha=alpha/2)
ax.plot(x, Plist[4], color='black') # Plot the original signal
#################################### one day case
# for i, val in enumerate(vals):
#     alpha = 0.5*(i+1)/len(vals) # Modify the alpha value for each iteration.
#     if i == 0:
#         ax.fill_between(oneday_x, Plist[8][0:24], Plist[0][0:24], color='red', alpha=alpha)
#     elif i == 1:
#         ax.fill_between(oneday_x, Plist[7][0:24], Plist[1][0:24], color='red', alpha=alpha)
#     elif i == 2:
#         ax.fill_between(oneday_x, Plist[6][0:24], Plist[2][0:24], color='red', alpha=alpha)
#     else:
#         ax.fill_between(oneday_x, Plist[5][0:24], Plist[3][0:24], color='red', alpha=alpha)
# ax.plot(oneday_x, Plist[4][0:24], color='black') # Plot the original signal

pl.show()

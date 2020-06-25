import sys
import pandas as pd
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1, type=float)
parser.add_argument('--iter', default=4000, type=int)
parser.add_argument('--beta1', default=0.9, type=float)
parser.add_argument('--beta2', default=0.999, type=float)
parser.add_argument('--weight-decay', default=1e-4, type=float)
parser.add_argument('--num_feat', default=18, type=int)
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--comment', default='', type=str)
args = parser.parse_args()

data = pd.read_csv('data/train.csv', encoding = 'big5')
data = data.iloc[:, 3:]
data[data == 'NR'] = 0
raw_data = data.to_numpy()

month_data = {}
for month in range(12):
    sample = np.empty([18, 480])
    for day in range(20):
        sample[:, day * 24 : (day + 1) * 24] = raw_data[18 * (20 * month + day) : 18 * (20 * month + day + 1), :]
    month_data[month] = sample

x = np.empty([12 * 471, 18 * 9], dtype = float)
y = np.empty([12 * 471, 1], dtype = float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day == 19 and hour > 14:
                continue
            x[month * 471 + day * 24 + hour, :] = month_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1, -1) #vector dim:18*9 (9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9 9)
            y[month * 471 + day * 24 + hour, 0] = month_data[month][9, day * 24 + hour + 9] #value

# y /= 100
num_feat = args.num_feat
dim = num_feat * 9 + 1
# print(dim)
# w = np.ones([dim, 1]) / dim
# w = np.random.uniform(0, 0.01, size=[dim, 1])
w = np.zeros([dim, 1])

mean_x = np.mean(x, axis = 0) #18 * 9 
std_x = np.std(x, axis = 0) #18 * 9 
x = (x - mean_x) / std_x

x = np.concatenate((np.ones([12 * 471, 1]), x), axis = 1).astype(float)
np.random.seed(args.seed)


data = np.concatenate((x, y), axis=1)
np.random.shuffle(data)
import copy
x = copy.deepcopy(data[:, :-1])
y = copy.deepcopy(data[:, -1:])

import math
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
# print(x.shape, y.shape) 
# exit()

learning_rate = args.lr
iter_time = args.iter
beta1 = args.beta1
beta2 = args.beta2
weight_decay = args.weight_decay


adagrad = np.zeros([dim, 1])
momentum = np.zeros([dim, 1])
rms = np.zeros([dim, 1])
min_loss = 1e10
best_w = np.zeros([dim, 1])

def lr_scheduler(t):
    if t < 5000:
        return learning_rate
    elif t < 10000:
        return learning_rate * 0.32
    else:
        return learning_rate * 0.1

eps = 1e-10
for t in range(iter_time):
    # loss = np.sqrt(np.sum(np.power(np.dot(x, w) - y, 2))/471/12)#rmse
    lr = lr_scheduler(t)
    # weight = w
    # weight[0] = 0
    gradient = 2 * np.dot(x_train_set.T, np.dot(x_train_set, w) - y_train_set)
    # gradient[-1] = np.sum(gradient[:-1]) # bias gradient fix ??
    momentum = beta1 * momentum + (1 - beta1) * gradient
    rms = beta2 * rms + (1 - beta2) * gradient ** 2
    t1 = 1 - pow(beta1, t + 1)
    t2 = 1 - pow(beta2, t + 1)
    mcorr = momentum / t1
    scorr = rms / t2
    w = w - lr * mcorr / np.sqrt(scorr + eps)
    # print(gradient.shape)
    # exit()
    # adagrad += gradient ** 2
    # w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
    valid_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/len(x_validation))
    train_loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/len(x_train_set))
    if valid_loss < min_loss:
        min_loss = valid_loss
        best_w = w
        # print('Update, {}'.format(loss))
    if((t+1)%5==0):
        print('[Train Loss: {:.5}] [Valid Loss: {:.5}]'.format(
            train_loss, valid_loss
        ))
np.save('weight.npy', best_w)
print('best loss: {}'.format(min_loss))
with open('logs/adam.log', 'a') as log_file:
    log_file.write('lr: {}, iter: {}, beta: ({}, {}), min_loss: {}, seed: {}\n'.format(
        learning_rate, iter_time, beta1, beta2, min_loss, args.seed
    ))
# print(w, best_w)

# testdata = pd.read_csv('gdrive/My Drive/hw1-regression/test.csv', header = None, encoding = 'big5')
testdata = pd.read_csv('data/test.csv', header = None, encoding = 'big5')
test_data = testdata.iloc[:, 2:]
test_data[test_data == 'NR'] = 0
test_data = test_data.to_numpy()
test_x = np.empty([240, 18*9], dtype = float)
for i in range(240):
    test_x[i, :] = test_data[18 * i: 18* (i + 1), :].reshape(1, -1)
# test_x = np.concatenate((test_x ** 2, test_x), axis = 1).astype(float)

test_x = (test_x - mean_x) / std_x
test_x = np.concatenate((np.ones([240, 1]), test_x), axis = 1).astype(float)

w = np.load('weight.npy')
ans_y = np.dot(test_x, w)
ans_y[ans_y<0] = 0

import csv
with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)

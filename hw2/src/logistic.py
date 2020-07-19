import numpy as np
import pandas as pd
import argparse
# from util import DNNDataset, DNN, TestDataset
# from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--epoch', default=30, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--dim', default=100, type=int)
parser.add_argument('--seed', default=0, type=int)
# parser.add_argument('--semi_model_file', default='model/semi-clf.pt', type=str)
args = parser.parse_args()

np.random.seed(args.seed)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './submit.csv'
semi_output_fpath = './semi_submit.csv'

# Parse csv files to numpy array
# def read_data():
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = np.float32)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = np.float32)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = np.float32)
    # return X_train, Y_train, X_test
# X_train, Y_train, X_test = read_data()

def _normalize(X, train = True, specified_column = None, X_mean = None, X_std = None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column] ,0).reshape(1, -1)
        X_std  = np.std(X[:, specified_column], 0).reshape(1, -1)

    X[:,specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)
     
    return X, X_mean, X_std

def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    data = np.concatenate((X, np.expand_dims(Y, axis=1)), axis=1)
    np.random.shuffle(data)
    return data[:train_size, :-1], data[:train_size, -1], data[train_size:, :-1], data[train_size:, -1]

# dim = args.dim

# selector = SelectKBest(score_func=chi2, k=dim).fit(X_train, Y_train)
# pca = PCA(n_components=dim, random_state=args.seed).fit(X_train)
# X_train = selector.transform(X_train)
# X_test = selector.transform(X_test)

dim = X_train.shape[1]
not_one_hot = []
for i in range(dim):
    feat = X_train[:, i]
    min, max = np.min(feat), np.max(feat)
    if not (min == 0.0 and max == 1.0):
        not_one_hot.append(i)

X_tr = X_train.T[not_one_hot]
X_te = X_test.T[not_one_hot]
X_train = np.concatenate((X_train, np.power(X_tr, 0.01).T, np.power(X_tr, 0.7).T, np.power(X_tr, 0.1).T, np.power(X_tr, 0.5).T), axis=1)
X_test = np.concatenate((X_test, np.power(X_te, 0.01).T, np.power(X_te, 0.7).T, np.power(X_te, 0.1).T, np.power(X_te, 0.5).T), axis=1)
X_train, X_mean, X_std = _normalize(X_train, train = True)
X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)


# Split data into training set and development set
dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)

train_size = X_train.shape[0]
dev_size = X_dev.shape[0]
test_size = X_test.shape[0]
data_dim = X_train.shape[1]
print('Size of training set: {}'.format(train_size))
print('Size of development set: {}'.format(dev_size))
print('Size of testing set: {}'.format(test_size))
print('Dimension of data: {}'.format(data_dim))
# data_dim = args.dim
# exit()

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _sigmoid(z):
    # Sigmoid function can be used to calculate probability.
    # To avoid overflow, minimum/maximum output value is set.
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))

def _f(X, w, b):
    # This is the logistic regression function, parameterized by w and b
    #
    # Arguements:
    #     X: input data, shape = [batch_size, data_dimension]
    #     w: weight vector, shape = [data_dimension, ]
    #     b: bias, scalar
    # Output:
    #     predicted probability of each row of X being positively labeled, shape = [batch_size, ]
    return _sigmoid(np.matmul(X, w) + b)

def _predict(X, w, b):
    # This function returns a truth value prediction for each row of X 
    # by rounding the result of logistic regression function.
    return np.round(_f(X, w, b)).astype(np.int)
    
def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc  

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label, np.log(y_pred)) - np.dot((1 - Y_label), np.log(1 - y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad

# Zero initialization for weights ans bias
w = np.zeros((data_dim,))
b = np.zeros((1,))
best_w = np.zeros((data_dim,)) 

# Some parameters for training    
max_iter = args.epoch
batch_size = args.batch_size
learning_rate = args.lr
eps = 1e-8

# Keep the loss and accuracy at every iteration for plotting
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []

max_acc = 0
# Calcuate the number of parameter updates
step = 1

# Iterative training
for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)
        
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):
        X = X_train[idx*batch_size:(idx+1)*batch_size]
        Y = Y_train[idx*batch_size:(idx+1)*batch_size]

        # Compute the gradient
        w_grad, b_grad = _gradient(X, Y, w, b)

        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad

        step = step + 1
            
    # Compute loss and accuracy of training set and development set
    y_train_pred = _f(X_train, w, b)
    Y_train_pred = np.round(y_train_pred)
    train_acc.append(_accuracy(Y_train_pred, Y_train))
    train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

    y_dev_pred = _f(X_dev, w, b)
    Y_dev_pred = np.round(y_dev_pred)
    dev_acc.append(_accuracy(Y_dev_pred, Y_dev))
    dev_loss.append(_cross_entropy_loss(y_dev_pred, Y_dev) / dev_size)
    if dev_acc[-1] > max_acc:
        best_w = w
        max_acc = dev_acc[-1]

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(np.min(dev_loss[-1])))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(np.max(dev_acc[-1])))

print('Use max dev acc model')
w = best_w
# X_test = pca.transform(X_test)

# Predict testing labels
predictions = _predict(X_test, w, b)
with open(output_fpath, 'w') as f:
    f.write('id,label\n')
    for i, label in  enumerate(predictions):
        f.write('{},{}\n'.format(i, label))

with open('logs/logistic.log', 'a') as log_file:
    log_file.write('lr: {}, iter: {}, max_acc: {}, seed: {}\n'.format(
        learning_rate, max_iter, np.max(dev_acc), args.seed
    ))

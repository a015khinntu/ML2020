import numpy as np
import pandas as pd


np.random.seed(0)
X_train_fpath = './data/X_train'
Y_train_fpath = './data/Y_train'
X_test_fpath = './data/X_test'
output_fpath = './submit.csv'

# Parse csv files to numpy array
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = float)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = float)

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

# X_train, X_mean, X_std = _normalize(X_train, train = True)
# X_test, _, _= _normalize(X_test, train = False, specified_column = None, X_mean = X_mean, X_std = X_std)

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


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
clf = GradientBoostingClassifier(n_estimators=150, verbose=1, max_depth=4)
clf.fit(X_train, Y_train)
y_pred = clf.predict(X_train)
y_dpred = clf.predict(X_dev)
train_acc = accuracy_score(Y_train, y_pred)
dev_acc = accuracy_score(Y_dev, y_dpred)
print('Train acc: {:.5}, Dev acc: {:.5}'.format(
    train_acc, dev_acc
))

predict
pred = clf.predict(X_test).flatten().astype(int)
index = np.arange(len(X_test))
df = pd.DataFrame({'id':index, 'label':pred})
df.to_csv(output_fpath, index=False)

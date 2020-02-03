import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn
from functools import partial
from torch._six import inf
import torch
from math import pi

feats = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
         'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR'
        ]

pm25 = 9
direc = 15

def mse(y_pred, y_true):
    return torch.mean((y_pred - y_true) ** 2).item()

class DNN(nn.Module):
    def __init__(self, dropout = 0.1, num_feat=162, hidden_dim=256):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(num_feat, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(self.dropout(self.relu(self.bn1(x))))
        x = self.l3(self.dropout(self.relu(self.bn2(x))))
        x = self.l4(self.dropout(self.relu(self.bn3(x))))
        x = self.act(x)
        return x

class EmbeddingNet(nn.Module):
    def __init__(self, hidden_dim = 512, num_embed = 10, embed_dim = 5, dropout = 0.1):
        super(EmbeddingNet, self).__init__()
        self.wind_direc = nn.Embedding(num_embed, embed_dim)
        self.direc_transfrom = nn.Linear(embed_dim * 9, embed_dim * 9)
        self.l1 = nn.Linear(153, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim + embed_dim * 9)
        self.l2 = nn.Linear(hidden_dim + embed_dim * 9, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.l3 = nn.Linear(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.l4 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.act = nn.Sigmoid()

    def forward(self, feature, direc):
        batch_size = feature.shape[0]
        feature = self.l1(feature)
        direc = self.wind_direc(direc)
        direc = self.direc_transfrom(direc.view(batch_size, -1))
        feature = torch.cat((feature, direc), dim=-1)
        feature = self.l2(self.dropout(self.relu(self.bn1(feature))))
        feature = self.l3(self.dropout(self.relu(self.bn2(feature))))
        feature = self.l4(self.dropout(self.relu(self.bn3(feature))))
        return feature

class DNNDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def get_direc(x_train):
    x_train = x_train.view(-1, 18, 9)
    x_new = torch.cat((x_train[:, :direc, :], x_train[:, direc + 1:, :]), dim=1)
    directions = x_train[:, direc, :]
    return x_new.view(-1, 153), directions.long()

def read_train_v0(fn='train.csv'):
    ''''NR' = 0, return (num_samples, 18, 9)'''
    df = pd.read_csv(fn, encoding='big5')
    data = df.values[:, 3:]
    num_feat = 18
    num_days = 20
    num_month = 12
    train_data = np.zeros((num_month, num_feat, num_days * 24), dtype=object)
    for i in range(num_month):
        for j in range(num_days):
            data_id = (i * num_month + j) * 18
            train_id = j * 24
            train_data[i, :, train_id: train_id + 24] = data[data_id: data_id + num_feat, :]

    train_data[train_data=='NR'] = 0
    train_data = train_data.astype(np.float32)
    train_data[train_data==float('nan')] = 0

    return train_data

def read_train_v1(fn='train.csv'):
    '''[train < 0] = 0, return (num_samples, 18, 9)'''
    train_data = read_train_v0(fn)
    train_data[train_data < 0] = 0
    return train_data

def read_train_v2(fn='train.csv'):
    train_data = read_train_v0(fn)
    train_data[:, direc, :] = np.sinh(train_data[:, direc, :] / 180 * pi)
    return train_data


def read_test_v0(fn='test.csv'):
    '''return (num_samples, 162)'''
    df = pd.read_csv(fn, header=None)
    data = df.values[:, 2:]
    data[data=='NR'] = 0
    return data.astype(np.float32).reshape(-1, 162)

def read_test_v1(fn='test.csv'):
    '''[test < 0] = 0 return (num_samples, 162)'''
    data = read_test_v0(fn)
    data[data < 0] = 0
    return data

def read_test_v2(fn='test.csv'):
    data = read_test_v0(fn)
    data = data.reshape(-1, 18, 9)
    data[:, direc, :] = np.sinh(data[:, direc, :] / 180 * pi)
    return data.reshape(-1, 162)

def extract_train_LR_v0(train_data, num_hour=9):
    '''
    return x: (num_samples, 162), y: (num_samples)
    '''
    hour = train_data.shape[2] - num_hour - 1
    x_train = []
    y_train = []
    count = 0
    for month in range(12):
        for h in range(hour):
            x_train.append(train_data[month, :, h:h+9].flatten())
            y_train.append(train_data[month][pm25][h+9])

    return np.array(x_train, dtype=np.float32),  np.array(y_train, dtype=np.float32).flatten()

def extract_train_LR_v1(train_data, num_hour=9):
    '''
    discard negtive samples
    return x: (num_samples, 162), y: (num_samples)
    '''
    hour = train_data.shape[2] - num_hour - 1
    x_train = []
    y_train = []
    count = 0
    for month in range(12):
        for h in range(hour):
            if min(train_data[month, :, h:h+9].flatten()) >= 0 and train_data[month][pm25][h+9] >= 0:
                x_train.append(train_data[month, :, h:h+9].flatten())
                y_train.append(train_data[month][pm25][h+9])

    return np.array(x_train, dtype=np.float32),  np.array(y_train, dtype=np.float32).flatten()

def process_embed(x, num_embed):
    x = x.reshape(-1, 18, 9)
    directions = x[:, direc, :].astype(int) // (360 // num_embed)
    x[:, direc, :] = directions
    return x.reshape(-1, 162)

def extract_train_embed(train_data, num_embed):
    x, y = extract_train_LR_v0(train_data)
    x = process_embed(x, num_embed)
    return x, y

def random_split(x, y, split_rate=0.2):
    train = np.concatenate((x, y), axis=-1)
    np.random.shuffle(train)
    length = int(len(train) * split_rate)
    valid = train[:length]
    train = train[length:]
    x_train = train[:, :-1]
    y_train = train[:, -1:]
    x_valid = valid[:, :-1]
    y_valid = valid[:, -1:]
    return (x_train, y_train), (x_valid, y_valid)

class ReduceLR(object):
    '''
    ReduceLROnPlateau from torch.optim.lr_scheduler
    '''

    def __init__(self, lr, mode='min', factor=0.1, patience=10,
                 verbose=True, threshold=1e-4, threshold_mode='rel',
                 cooldown=0, min_lr=0, eps=1e-8):

        if factor >= 1.0:
            raise ValueError('Factor should be < 1.0.')
        self.factor = factor

        if not isinstance(lr, float) and not isinstance(lr, int):
            raise TypeError('{} is not a number'.format(
                type(lr).__name__))
        self.lr = lr

        self.min_lr = min_lr

        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0
        self.mode = mode
        self.best = None
        self.num_bad_epochs = None
        self.mode_worse = None  # the worse value for the chosen mode
        self.is_better = None
        self.eps = eps
        self.last_epoch = -1
        self._init_is_better(mode=mode, threshold=threshold,
                             threshold_mode=threshold_mode)
        self._reset()

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def step(self, metrics, epoch=None):
        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(metrics)
        if epoch is None:
            epoch = self.last_epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # ignore any bad epochs in cooldown

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0
        

    def _reduce_lr(self, epoch):
        old_lr = self.lr
        self.lr = max(old_lr * self.factor, self.min_lr)
        if self.verbose and old_lr != self.min_lr:
            print('Learning Rate reduce from {:.4} to {:.4}'.format(old_lr, self.lr))

    @property
    def in_cooldown(self):
        return self.cooldown_counter > 0

    def _cmp(self, mode, threshold_mode, threshold, a, best):
        if mode == 'min' and threshold_mode == 'rel':
            rel_epsilon = 1. - threshold
            return a < best * rel_epsilon

        elif mode == 'min' and threshold_mode == 'abs':
            return a < best - threshold

        elif mode == 'max' and threshold_mode == 'rel':
            rel_epsilon = threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + threshold

    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + threshold_mode + ' is unknown!')

        if mode == 'min':
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.is_better = partial(self._cmp, mode, threshold_mode, threshold)

'''
extract_train_LR_v1: discard negtive samples
'''

read_test = read_test_v2
read_train = read_train_v2
extract_train_LR = extract_train_LR_v0

if __name__ == "__main__":
    data = read_train()
    # read_test()
    x, y = extract_train_LR(data)
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
import torch.nn as nn

feats = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx', 'O3', 'PM10', 'PM2.5',
         'RAINFALL', 'RH', 'SO2', 'THC', 'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR'
        ]

pm25 = 9

class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.l1 = nn.Linear(162, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(self.relu(x))
        x = self.l3(self.relu(x))
        return x

class DNNDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

def read_train(fn='train.csv'):
    df = pd.read_csv(fn, encoding='big5')
    data = df.values[:, 3:]
    num_feat = 18
    num_days = 20
    num_month = 12
    train_data = np.zeros((num_month, num_feat, num_days * 24), dtype=object)
    for i in range(num_month):
        for j in range(num_days):
            data_id = (i * num_month + j) * 18
            # print(data_id)
            train_id = j * 24
            train_data[i, :, train_id: train_id + 24] = data[data_id: data_id + num_feat, :]

    train_data[train_data=='NR'] = 0
    train_data = train_data.astype(np.float32)
    train_data[train_data==float('nan')] = 0

    return train_data

def extract_train_LR(train_data, num_hour=9):
    hour = train_data.shape[2] - num_hour - 1
    x_train = np.zeros((hour * 12, 18 * num_hour), dtype=float)
    y_train = np.zeros((hour * 12, 1), dtype=float)
    count = 0
    for month in range(12):
        for h in range(hour):
            x_train[count] = train_data[month, :, h:h+9].flatten()
            y_train[count] = train_data[month][pm25][h+9]
            count += 1

    return x_train, y_train

def read_test(fn='test.csv'):
    df = pd.read_csv(fn, header=None)
    data = df.values[:, 2:]
    data[data=='NR'] = 0
    return data.astype(np.float32).reshape(-1, 18, 9)

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

if __name__ == "__main__":
    data = read_train()
    # read_test()
    x, y = extract_train_LR(data)
import numpy as np
import pandas as pd
import argparse
from util import DNNDataset, DNN, TestDataset

parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('--hidden_dim', default=128, type=int)
parser.add_argument('--num_layers', default=3, type=int)
parser.add_argument('--epoch', default=10, type=int)
parser.add_argument('--batch_size', default=32, type=int)
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
with open(X_train_fpath) as f:
    next(f)
    X_train = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = np.float32)
with open(Y_train_fpath) as f:
    next(f)
    Y_train = np.array([line.strip('\n').split(',')[1] for line in f], dtype = np.float32)
with open(X_test_fpath) as f:
    next(f)
    X_test = np.array([line.strip('\n').split(',')[1:] for line in f], dtype = np.float32)

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

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
# from tqdm import tqdm



def compute_acc(y_pred, y_true):
    score = torch.tensor([1 if sc.item() >= 0.5 else 0 for sc in y_pred])
    return torch.sum(torch.eq(score, y_pred.cpu().long())).item() / len(y_true)

# train process
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = DNN(input_size=data_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
train_ds = DNNDataset(X_train, Y_train)
train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
valid_ds = DNNDataset(X_dev, Y_dev)
valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=True)

optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
criterion = nn.BCELoss()
max_acc = 0
for epoch in range(args.epoch):
    # train
    model.train()
    train_loss = train_acc = 0.0
    for batch in train_loader:
        x, y = (t.to(device) for t in batch)
        y_pred = model(x).squeeze()
        loss = criterion(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_acc += compute_acc(y_pred, y)
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    # valid
    model.eval()
    valid_loss = valid_acc = 0.0
    for batch in valid_loader:
        x, y = (t.to(device) for t in batch)
        y_pred = model(x).squeeze()
        loss = criterion(y_pred, y)
        valid_loss += loss.item()
        valid_acc += compute_acc(y_pred, y)
    valid_loss /= len(valid_loader)
    valid_acc /= len(valid_loader)
    print('[Epoch: {:04}] [Train Loss: {:.4}] [Train Acc: {:.4}] [Valid Loss: {:.4}] [Valid Acc: {:.4}]'.format(
        epoch + 1, train_loss, train_acc, valid_loss, valid_acc
    ))
    if valid_acc > max_acc:
        torch.save(model.state_dict(), 'model/clf-l{}-h{}.pt'.format(
            args.num_layers, args.hidden_dim
        ))
        max_acc = valid_acc

model.load_state_dict(torch.load('model/clf-l{}-h{}.pt'.format(
    args.num_layers, args.hidden_dim
)))

output = []
for batch in DataLoader(TestDataset(X_test), batch_size=args.batch_size):
    sent = batch.to(device)
    score = model(sent).squeeze()
    output.extend([1 if sc.item() >= 0.5 else 0 for sc in score])
df = pd.DataFrame({'label': output})
df.to_csv(output_fpath, index_label='id')

# if max_acc < 0.89:
#     print('Trained model is not that good, pause!')
#     exit()
# else:
#     print('Start use pseudo label to train model!')

# Y_test = np.array(output, dtype=np.float32)
# X_semi = np.concatenate((X_train, X_test), axis=0)
# Y_semi = np.concatenate((Y_train, Y_test), axis=0)
# semi_ds = DNNDataset(X_semi, Y_semi)
# semi_loader = DataLoader(semi_ds, batch_size=args.batch_size, shuffle=True)

# semi_model = DNN(input_size=data_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers).to(device)
# optimizer = Adam(semi_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
# criterion = nn.BCELoss()
# max_acc = 0
# for epoch in range(args.epoch):
#     # train
#     semi_model.train()
#     train_loss = train_acc = 0.0
#     for batch in semi_loader:
#         x, y = (t.to(device) for t in batch)
#         y_pred = semi_model(x).squeeze()
#         loss = criterion(y_pred, y)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#         train_acc += compute_acc(y_pred, y)
#     train_loss /= len(semi_loader)
#     train_acc /= len(semi_loader)
#     # valid
#     semi_model.eval()
#     valid_loss = valid_acc = 0.0
#     for batch in valid_loader:
#         x, y = (t.to(device) for t in batch)
#         y_pred = semi_model(x).squeeze()
#         loss = criterion(y_pred, y)
#         valid_loss += loss.item()
#         valid_acc += compute_acc(y_pred, y)
#     valid_loss /= len(valid_loader)
#     valid_acc /= len(valid_loader)
#     print('[Epoch: {:04}] [Train Loss: {:.4}] [Train Acc: {:.4}] [Valid Loss: {:.4}] [Valid Acc: {:.4}]'.format(
#         epoch + 1, train_loss, train_acc, valid_loss, valid_acc
#     ))
#     if valid_acc > max_acc:
#         torch.save(semi_model.state_dict(), 'model/semi-clf-l{}-h{}.pt'.format(
#             args.num_layers, args.hidden_dim
#         ))
#         max_acc = valid_acc

# semi_model.load_state_dict(torch.load('model/semi-clf-l{}-h{}.pt'.format(
#     args.num_layers, args.hidden_dim
# )))

# output = []
# for batch in DataLoader(TestDataset(X_test), batch_size=args.batch_size):
#     sent = batch.to(device)
#     score = semi_model(sent).squeeze()
#     output.extend([1 if sc.item() >= 0.5 else 0 for sc in score])
# df = pd.DataFrame({'label': output})
# df.to_csv(semi_output_fpath, index_label='id')

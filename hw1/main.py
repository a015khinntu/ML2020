import argparse
from sklearn.linear_model import LinearRegression
from util import *
import pandas as pd
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD, RMSprop, Adam, Adadelta, Adagrad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
import numpy as np

def apply_LR(args):
    data = read_train()
    x, y = extract_train_LR(data)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    y_pred[y_pred < 0] = 0
    mse = np.mean(np.power(y_pred - y, 2))
    print(mse)
    test = read_test()
    num_test = test.shape[0]
    test = test.reshape(num_test, -1)
    pred = model.predict(test)
    pred = pred.flatten()
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_nn_base(args):
    data = read_train()
    x, y = extract_train_LR(data)
    model = nn.Linear(162, 1)
    ds = DNNDataset(x.astype(np.float32), y.astype(np.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    min_loss = 1e10
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epoch):
        model.train()
        train_loss = valid_loss = 0.0
        for x_train, y_train in dl:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(dl)
        if train_loss < min_loss:
            torch.save(model, 'model/linear-base.pt')
            min_loss = train_loss
        print(train_loss)
    print(min_loss)
    model = torch.load('model/linear-base.pt')
    model.eval()
    test = read_test()
    num_test = test.shape[0]
    test = torch.tensor(test.reshape(num_test, -1))
    pred = model(test)
    pred = pred.detach().numpy().flatten()
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_normLR(args):
    data = read_train()
    x, y = extract_train_LR(data)
    max_feat = np.max(x, axis=0)
    max_y = np.max(y)
    x /= max_feat
    y /= max_y
    model = nn.Linear(162, 1)
    ds = DNNDataset(x.astype(np.float32), y.astype(np.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    min_loss = 1e10
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epoch):
        model.train()
        train_loss = valid_loss = 0.0
        for x_train, y_train in dl:
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(dl)
        if train_loss < min_loss:
            torch.save(model, 'model/linear-base.pt')
            min_loss = train_loss
        print(train_loss * max_y ** 2)
    model = torch.load('model/linear-base.pt')
    x = torch.tensor(x, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    for i in range(10):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
        print(loss.item() * max_y ** 2)
        if loss < min_loss:
            torch.save(model, 'model/linear-base.pt')
            min_loss = loss
    print(min_loss * max_y ** 2)
    model = torch.load('model/linear-base.pt')
    model.eval()
    test = read_test()
    num_test = test.shape[0]
    test = test.reshape(num_test, -1)
    test /= max_feat
    test = torch.tensor(test)
    pred = model(test)
    pred = pred.detach().numpy().flatten()
    pred[pred < 0] = 0
    pred *= max_y
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_dnn(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = read_train()
    x, y = extract_train_LR(data)
    max_feat = np.max(x, axis=0)
    max_y = np.max(y)
    x /= max_feat
    y /= max_y
    # exit()
    (x, y), (x_valid, y_valid) = random_split(x, y)
    x_valid = torch.tensor(x_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
    model = DNN().to(device)
    ds = DNNDataset(x.astype(np.float32), y.astype(np.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    min_loss = 1e10
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(args.epoch):
        model.train()
        train_loss = valid_loss = 0.0
        for batch in dl:
            x_train, y_train = (t.to(device) for t in batch)
            optimizer.zero_grad()
            y_pred = model(x_train)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = criterion(model(x_valid), y_valid).item()
        train_loss /= len(dl)
        if valid_loss < min_loss:
            torch.save(model, 'model/linear-base.pt')
            min_loss = valid_loss
        print(np.sqrt(train_loss * max_y ** 2), np.sqrt(valid_loss * max_y ** 2))
    
    print(np.sqrt(min_loss * max_y ** 2))
    model = torch.load('model/linear-base.pt')
    model.eval()
    test = read_test()
    num_test = test.shape[0]
    test = test.reshape(num_test, -1)
    test /= max_feat
    test = torch.tensor(test).to(device)
    pred = model(test)
    pred = pred.detach().cpu().numpy().flatten()
    pred[pred < 0] = 0
    pred *= max_y
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_mean(args):
    test = read_test()
    num_test = test.shape[0]
    test = test.reshape(num_test, 18, 9)[:, 9, :]
    pred = np.mean(test, axis=1)
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_last(args):
    test = read_test()
    num_test = test.shape[0]
    pred = test.reshape(num_test, 18, 9)[:, 9, -1]
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def main(args):
    if args.mode == 'LR':
        apply_LR(args)
    elif args.mode == 'nn-base':
        apply_nn_base(args)
    elif args.mode == 'normLR':
        apply_normLR(args)
    elif args.mode == 'dnn':
        apply_dnn(args)
    elif args.mode == 'mean':
        apply_mean(args)
    elif args.mode == 'last':
        apply_last(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='submission.csv', type=str)
    parser.add_argument('--mode', default='LR', type=str)
    parser.add_argument('--lr', default=1e-6, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epoch', default=20, type=int)
    parser.add_argument('--weight-decay', default=1e-4, type=float)
    args = parser.parse_args()
    main(args)

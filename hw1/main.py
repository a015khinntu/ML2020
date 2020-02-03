import argparse
from sklearn.linear_model import LinearRegression
from util import *
import pandas as pd
import torch.nn as nn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim import SGD, RMSprop, Adam, Adadelta, Adagrad
from torch.utils.data import DataLoader
import numpy as np
from sklearn.ensemble import RandomForestClassifier



def apply_LR(args):
    data = read_train()
    x, y = extract_train_LR(data)
    model = LinearRegression()
    model.fit(x, y)
    y_pred = model.predict(x)
    y_pred[y_pred < 0] = 0
    mse = np.mean(np.power(y_pred - y, 2))
    print(np.sqrt(mse))
    test = read_test()
    pred = model.predict(test)
    pred = pred.flatten()
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_nn_base(args):
    data = read_train()
    x, y = extract_train_LR(data)
    y = np.expand_dims(y, 1)
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
            torch.save(model, 'model/nn-base.pt')
            min_loss = train_loss
        print(train_loss)
    print(min_loss)
    model = torch.load('model/nn-base.pt')
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
    y = np.expand_dims(y, 1)
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
            torch.save(model, 'model/normLR.pt')
            min_loss = train_loss
        if (epoch + 1) % args.sample_step == 0:
            print('[Epoch: {:5}] [Train Loss: {:.4}] [Valid Loss: {:.4}]'.format(
                    epoch + 1, np.sqrt(train_loss * max_y ** 2), np.sqrt(valid_loss * max_y ** 2))
                )

    print(np.sqrt(min_loss * max_y ** 2))
    model = torch.load('model/normLR.pt')
    model.eval()
    test = read_test()
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
    y = np.expand_dims(y, 1)
    max_feat = np.max(x, axis=0)
    max_y = np.max(y)
    x /= max_feat
    y /= max_y
    # exit()
    (x, y), (x_valid, y_valid) = random_split(x, y)
    x_valid = torch.tensor(x_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
    model = DNN(dropout=args.dropout, num_feat=args.num_feat * 9, hidden_dim=args.hidden_dim).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    ds = DNNDataset(x.astype(np.float32), y.astype(np.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    min_loss = 1e10
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # writer = SummaryWriter('logs/')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, cooldown=50, min_lr=1e-8)
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
        scheduler.step(valid_loss)
        train_loss /= len(dl)
        if valid_loss < min_loss:
            torch.save(model, 'model/dnn.pt')
            min_loss = valid_loss
        if (epoch + 1) % args.sample_step == 0:
            print('[Epoch: {:5}] [Train Loss: {:.4}] [Valid Loss: {:.4}]'.format(
                    epoch + 1, np.sqrt(train_loss * max_y ** 2), np.sqrt(valid_loss * max_y ** 2))
                )
        # writer.add_scalar('train_loss', np.sqrt(train_loss * max_y ** 2), global_step=epoch)
        # writer.add_scalar('valid_loss', np.sqrt(valid_loss * max_y ** 2), global_step=epoch)
    
    # writer.close()
    print(np.sqrt(min_loss * max_y ** 2))
    model = torch.load('model/dnn.pt')
    model.eval()
    test = read_test()
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
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_last(args):
    test = read_test()
    num_test = test.shape[0]
    pred = test.reshape(num_test, 18, 9)[:, 9, -1]
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_clf_LR(args):
    data = read_train()
    x, y = extract_train_LR(data)
    month_x = x.reshape(12, -1, 162)
    month_y = y.reshape(12, -1, 1)
    num_month_data = month_x.shape[1]
    month = (np.ones((num_month_data, 1)) * np.arange(12)).T.flatten()
    months_model = [LinearRegression() for _ in range(12)]
    for i in range(12):
        mx = month_x[i]
        my = month_y[i]
        months_model[i].fit(mx, my)
    clf = RandomForestClassifier()
    clf.fit(x, month)
    test = read_test()
    labels = clf.predict(test).astype(int)
    pred = []
    for i, l in enumerate(labels):
        value = months_model[l].predict([test[i]])
        pred.append(value[0][0])
    pred = np.array(pred)
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_regressor(args):
    from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVR
    regressor = LinearSVR()
    data = read_train()
    x, y = extract_train_LR(data)
    regressor.fit(x, y.flatten())
    test = read_test()
    pred = regressor.predict(test).flatten()
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_feat_select(args):
    from sklearn.feature_selection import SelectKBest, f_regression
    data = read_train()
    x, y = extract_train_LR(data)
    selector = SelectKBest(f_regression, args.num_feat * 9)
    x_new = selector.fit_transform(x, y)
    regressor = LinearRegression()
    regressor.fit(x_new, y)
    test = read_test()
    test_new = selector.transform(test)
    pred = regressor.predict(test_new).flatten()
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_fs_nn(args):
    from sklearn.feature_selection import SelectKBest, f_regression
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = read_train()
    x, y = extract_train_LR(data)
    y = np.expand_dims(y, 1)
    selector = SelectKBest(f_regression, args.num_feat * 9)
    x_new = selector.fit_transform(x, y)
    max_feat = np.max(x_new, axis=0)
    max_y = np.max(y)
    x_new /= max_feat
    y /= max_y
    # exit()
    (x, y), (x_valid, y_valid) = random_split(x_new, y)
    x_valid = torch.tensor(x_valid, dtype=torch.float32).to(device)
    y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
    model = DNN(dropout=args.dropout, num_feat=args.num_feat * 9, hidden_dim=args.hidden_dim).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    ds = DNNDataset(x.astype(np.float32), y.astype(np.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    min_loss = 1e10
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # writer = SummaryWriter('logs/')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=50, cooldown=50, min_lr=1e-6, verbose=True)
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
            train_loss += mse(y_pred, y_train)

        model.eval()
        valid_loss = mse(model(x_valid), y_valid)
        scheduler.step(valid_loss)
        train_loss /= len(dl)
        if valid_loss < min_loss:
            torch.save(model, 'model/fs_nn.pt')
            min_loss = valid_loss
        if (epoch + 1) % args.sample_step == 0:
            print('[Epoch: {:5}] [Train Loss: {:.4}] [Valid Loss: {:.4}]'.format(
                    epoch + 1, np.sqrt(train_loss * max_y ** 2), np.sqrt(valid_loss * max_y ** 2))
                )
        # writer.add_scalar('train_loss', np.sqrt(train_loss * max_y ** 2), global_step=epoch)
        # writer.add_scalar('valid_loss', np.sqrt(valid_loss * max_y ** 2), global_step=epoch)
    
    # writer.close()
    print(np.sqrt(min_loss * max_y ** 2))
    model = torch.load('model/fs_nn.pt')
    model.eval()
    test = read_test()
    test_new = selector.transform(test)
    test_new /= max_feat
    test_new = torch.tensor(test_new).to(device)
    pred = model(test_new)
    pred = pred.detach().cpu().numpy().flatten()
    pred[pred < 0] = 0
    pred *= max_y
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_np_LR(args):
    '''
    hand writr gradient descent with adam optimizer, regularization
    '''
    lr = args.lr
    data = read_train()
    x, y = extract_train_LR(data)
    max_feat = np.max(x, axis=0)
    max_y = np.max(y)
    x = x / max_feat
    y = y / max_y
    best_model = np.ones(163)
    # weight = np.random.uniform(0, 0.01, 162)
    weight = np.zeros(162)
    bias = 0.0
    lda = args.weight_decay
    episilon = 1e-8 # as in keras eposilon = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    momentum = np.zeros(163)
    rms = np.zeros(163)
    Loss = []
    min_rms = 1e10
    scheduler = ReduceLR(lr, mode='min', factor=0.1, patience=200, cooldown=200, min_lr=1e-8)
    for epoch in range(args.epoch):
        y_predict = x @ weight + bias
        Loss = y_predict - y
        Wgradient = 2 * x.T @ Loss + weight * lda
        Bgradient = 2 * sum(Loss)
        momentum[:-1] = beta1 * momentum[:-1] + (1 - beta1) * Wgradient
        momentum[-1] = beta1 * momentum[-1] + (1 - beta1) * Bgradient
        rms[:-1] = beta2 * rms[:-1] + (1 - beta2) * (Wgradient**2)
        rms[-1] = beta2 * rms[-1] + (1 - beta2) * (Bgradient**2)
        timeRitio1 = 1 - pow(beta1, epoch + 1)
        timeRitio2 = 1 - pow(beta2, epoch + 1)
        Mcorr = momentum / timeRitio1
        Scorr = rms / timeRitio2
        weight -= scheduler.lr * Mcorr[:-1] / (np.sqrt(Scorr[:-1]) + episilon)
        bias -= scheduler.lr * Mcorr[-1] / (np.sqrt(Scorr[-1]) + episilon)
        rmsValue = np.sqrt(np.sum(Loss ** 2) / len(x)) * max_y
        scheduler.step(rmsValue)
        if rmsValue < min_rms:
            best_model[:-1] = weight
            best_model[-1] = bias
            np.savetxt('model/nplr.txt', best_model)
            min_rms = rmsValue
        if (epoch + 1) % args.sample_step == 0:
            print('[Epoch {:4}] [Train Loss {:.4}]'.format(
                epoch + 1, rmsValue
            ))

    print(min_rms)
    test = read_test() / max_feat
    weight = best_model[:-1]
    bias = best_model[-1]
    pred = (test @ weight + bias) * max_y
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_fs_np_LR(args):
    '''
    hand writr gradient descent with adam optimizer, regularization
    '''
    from sklearn.feature_selection import SelectKBest, f_regression
    lr = args.lr
    data = read_train()
    x, y = extract_train_LR(data)
    selector = SelectKBest(f_regression, args.num_features)
    x_new = selector.fit_transform(x, y)
    max_feat = np.max(x_new, axis=0)
    max_y = np.max(y)
    x_new = x_new / max_feat
    y = y / max_y
    best_model = np.ones(args.num_features + 1)
    weight = np.random.uniform(0, 0.01, args.num_features)
    # weight = np.zeros(162)
    bias = 0.0
    lda = args.weight_decay
    episilon = 1e-8 # as in keras eposilon = 1e-8
    beta1 = 0.9
    beta2 = 0.999
    momentum = np.zeros(args.num_features + 1)
    rms = np.zeros(args.num_features + 1)
    Loss = []
    min_rms = 1e10
    scheduler = ReduceLR(lr, mode='min', factor=0.1, patience=200, cooldown=200, min_lr=1e-6)
    for epoch in range(args.epoch):
        y_predict = x_new @ weight + bias
        Loss = y_predict - y
        Wgradient = 2 * x_new.T @ Loss + weight * lda
        Bgradient = 2 * sum(Loss)
        momentum[:-1] = beta1 * momentum[:-1] + (1 - beta1) * Wgradient
        momentum[-1] = beta1 * momentum[-1] + (1 - beta1) * Bgradient
        rms[:-1] = beta2 * rms[:-1] + (1 - beta2) * (Wgradient**2)
        rms[-1] = beta2 * rms[-1] + (1 - beta2) * (Bgradient**2)
        timeRitio1 = 1 - pow(beta1, epoch + 1)
        timeRitio2 = 1 - pow(beta2, epoch + 1)
        Mcorr = momentum / timeRitio1
        Scorr = rms / timeRitio2
        weight -= scheduler.lr * Mcorr[:-1] / (np.sqrt(Scorr[:-1]) + episilon)
        bias -= scheduler.lr * Mcorr[-1] / (np.sqrt(Scorr[-1]) + episilon)
        rmsValue = np.sqrt(np.sum(Loss ** 2) / len(x)) * max_y
        scheduler.step(rmsValue)
        if rmsValue < min_rms:
            best_model[:-1] = weight
            best_model[-1] = bias
            np.savetxt('model/nplr.txt', best_model)
            min_rms = rmsValue
        if (epoch + 1) % args.sample_step == 0:
            print('[Epoch {:4}] [Train Loss {:.4}]'.format(
                epoch + 1, rmsValue
            ))

    print(min_rms)
    test = read_test()
    test_new = selector.transform(test) / max_feat
    weight = best_model[:-1]
    bias = best_model[-1]
    pred = (test_new @ weight + bias) * max_y
    pred[pred < 0] = 0
    index = ['id_{}'.format(i) for i in range(len(pred))]
    df = pd.DataFrame({'id':index, 'value':pred})
    df.to_csv(args.output, index=False)

def apply_embedding(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = read_train()
    x, y = extract_train_embed(data, args.num_embeddings)
    max_feat = np.max(x, axis=0).reshape(18, 9)
    max_feat = np.concatenate((max_feat[:15], max_feat[16:]), axis=0).flatten()
    max_feat = torch.tensor(max_feat, dtype=torch.float32).to(device)
    max_y = np.max(y)
    y /= max_y
    y = np.expand_dims(y, 1)
    (x, y), (x_valid, y_valid) = random_split(x, y)
    x_valid = torch.tensor(x_valid, dtype=torch.float32).to(device)
    x_valid, valid_direc = get_direc(x_valid)
    x_valid /= max_feat
    y_valid = torch.tensor(y_valid, dtype=torch.float32).to(device)
    model = EmbeddingNet(hidden_dim=args.hidden_dim, num_embed=args.num_embeddings, 
                         embed_dim=args.embedding_dim, dropout=args.dropout).to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))
    ds = DNNDataset(x.astype(np.float32), y.astype(np.float32))
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True)
    min_loss = 1e10
    criterion = nn.MSELoss()
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # writer = SummaryWriter('logs/')
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, cooldown=10, min_lr=1e-6, verbose=True)
    for epoch in range(args.epoch):
        model.train()
        train_loss = valid_loss = 0.0
        for batch in dl:
            x_train, y_train = (t.to(device) for t in batch)
            x_train, direc = get_direc(x_train)
            x_train /= max_feat
            optimizer.zero_grad()
            y_pred = model(x_train, direc)
            loss = criterion(y_pred, y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        valid_loss = criterion(model(x_valid, valid_direc), y_valid).item()
        scheduler.step(valid_loss)
        train_loss /= len(dl)
        if valid_loss < min_loss:
            torch.save(model, 'model/embedding-net.pt')
            min_loss = valid_loss
        if (epoch + 1) % args.sample_step == 0:
            print('[Epoch: {:5}] [Train Loss: {:.4}] [Valid Loss: {:.4}]'.format(
                    epoch + 1, np.sqrt(train_loss) * max_y, np.sqrt(valid_loss) * max_y)
                )
        # writer.add_scalar('train_loss', np.sqrt(train_loss * max_y ** 2), global_step=epoch)
        # writer.add_scalar('valid_loss', np.sqrt(valid_loss * max_y ** 2), global_step=epoch)
    
    # writer.close()
    print(np.sqrt(min_loss * max_y ** 2))
    model = torch.load('model/embedding-net.pt')
    model.eval()
    test = read_test()
    test = process_embed(test, args.num_embeddings)
    test = torch.tensor(test).to(device)
    test, test_direc = get_direc(test)
    test /= max_feat
    pred = model(test, test_direc)
    pred = pred.detach().cpu().numpy().flatten()
    pred[pred < 0] = 0
    pred *= max_y
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
    elif args.mode == 'clfLR':
        apply_clf_LR(args)
    elif args.mode == 'regressor':
        apply_regressor(args)
    elif args.mode == 'fSelect':
        apply_feat_select(args)
    elif args.mode == 'fSelect-nn':
        apply_fs_nn(args)
    elif args.mode == 'np-LR':
        apply_np_LR(args)
    elif args.mode == 'fS-np-LR':
        apply_fs_np_LR(args)
    elif args.mode == 'embedding':
        apply_embedding(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='submission.csv', type=str)
    parser.add_argument('--mode', default='LR', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--batch_size', default=5640, type=int)
    parser.add_argument('--epoch', default=5000, type=int)
    parser.add_argument('--sample_step', default=100, type=int)
    parser.add_argument('--weight_decay', default=1e-3, type=float)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--num_feat', default=18, type=int)
    parser.add_argument('--num_features', default=162, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--num_embeddings', default=15, type=int)
    parser.add_argument('--embedding_dim', default=5, type=int)
    args = parser.parse_args()
    main(args)

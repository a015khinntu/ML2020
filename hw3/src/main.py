import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import time
import pickle
import argparse

from nn_util import Classifier_v3 as Classifier
from util import ImgDataset, WarmUp, UniformNoise, GaussionNoise

from tensorboardX import SummaryWriter

def train_val(model, train_loader, val_loader, optimizer, num_epoch, writer, warmup_step, scheduler=None, global_step=0, last_epoch=0):
    warmup_scheduler = WarmUp(optimizer, warmup_step)

    local_step = global_step
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
        
            train_pred = model(data[0].to(device))
            batch_loss = loss(train_pred, data[1].to(device))
            batch_loss.backward()
            if (local_step + 1) % accu_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                if local_step < warmup_step:
                    warmup_scheduler.step()
            

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
            local_step += 1
    
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                val_pred = model(data[0].to(device))
                batch_loss = loss(val_pred, data[1].to(device))

                val_acc += np.sum(np.argmax(val_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
                val_loss += batch_loss.item()

        
            writer.add_scalar('train/acc', train_acc/train_set.__len__(), epoch + last_epoch)
            writer.add_scalar('train/loss', train_loss/train_set.__len__(), epoch + last_epoch)
            writer.add_scalar('val/acc', val_acc/val_set.__len__(), epoch + last_epoch)
            writer.add_scalar('val/loss', val_loss/val_set.__len__(), epoch + last_epoch)

        if scheduler != None and local_step >= warmup_step:
            scheduler.step()
    return local_step + global_step, epoch

def train(model, train_loader, optimizer, num_epoch, writer, warmup_step, scheduler=None, global_step=0, last_epoch=0):
    warmup_scheduler = WarmUp(optimizer, warmup_step)

    local_step = global_step
    for epoch in range(num_epoch):
        train_acc = 0.0
        train_loss = 0.0
        val_acc = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
        
            train_pred = model(data[0].to(device))
            batch_loss = loss(train_pred, data[1].to(device))
            batch_loss.backward()
            if (local_step + 1) % accu_step == 0:
                optimizer.step()
                optimizer.zero_grad()
                if local_step < warmup_step:
                    warmup_scheduler.step()
            

            train_acc += np.sum(np.argmax(train_pred.cpu().data.numpy(), axis=1) == data[1].numpy())
            train_loss += batch_loss.item()
            local_step += 1

        writer.add_scalar('train/acc', train_acc/train_set.__len__(), epoch + last_epoch)
        writer.add_scalar('train/loss', train_loss/train_set.__len__(), epoch + last_epoch)

        if scheduler != None and local_step >= warmup_step:
            scheduler.step()
    return local_step + global_step, epoch



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    parser.add_argument('--accu_step', default=4, type=int)
    parser.add_argument('--epoch', default=450, type=int)
    parser.add_argument('--lr', default=1e-3, type=float)
    parser.add_argument('--warmup', default=1000, type=int)
    args = parser.parse_args()
    
    train_x, train_y, val_x, val_y, test_x = pickle.load(open('data.pkl', 'rb'))

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),                                    
        transforms.ToTensor(),
    ])

    batch_size = args.batch_size
    accu_step = args.accu_step

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Classifier().to(device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('\n=== start training, parameter total:{}, trainable:{}'.format(total, trainable))

    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=3e-4)

    warmup_step = args.warmup

    # curriculum
    # num_epoch = 100
    writer = SummaryWriter('noisy')
    # cur_step, cur_epoch = train(model, train_loader, optimizer, num_epoch, writer, warmup_step)

    # training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.1, hue=0.05, saturation=0.1),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.RandomChoice(
            [GaussionNoise(0, 0.01), UniformNoise(-0.01, 0.01)]
        )
    ])
    train_val_x = np.concatenate((train_x, val_x), axis=0)
    train_val_y = np.concatenate((train_y, val_y), axis=0)
    train_set = ImgDataset(train_val_x, train_val_y, train_transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    num_epoch = args.epoch
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    # scheduler = ExponentialLR(optimizer, 0.99)
    scheduler = StepLR(optimizer, 150)

    train(model, train_loader, optimizer, num_epoch, writer, warmup_step, scheduler)

    test_set = ImgDataset(test_x, transform=test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    
    model.eval()
    prediction = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            test_pred = model(data.to(device))
            test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
            for y in test_label:
                prediction.append(y)

    with open("predict.csv", 'w') as f:
        f.write('Id,Category\n')
        for i, y in  enumerate(prediction):
            f.write('{},{}\n'.format(i, y))

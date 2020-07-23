from torch.utils.data import DataLoader, Dataset
import torch
import os
import numpy as np
import cv2
import pickle
from torch.optim.lr_scheduler import _LRScheduler

class WarmUp(_LRScheduler):
    def __init__(self, optimizer, warmup_step, last_epoch=-1):
        self.warmup_step = warmup_step
        super(WarmUp, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * (self.last_epoch / self.warmup_step)
                for base_lr in self.base_lrs]

class ImgDataset(Dataset):
    def __init__(self, x, y=None, transform=None):
        self.x = x
        # label is required to be a LongTensor
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        X = self.x[index]
        if self.transform is not None:
            X = self.transform(X)
        if self.y is not None:
            Y = self.y[index]
            return X, Y
        else:
            return X

class GaussionNoise(object):
    def __init__(self, mean = 0, std = 0.1):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        noise = torch.randn_like(sample)
        return noise * self.std + self.mean + sample

class UniformNoise(object):
    def __init__(self, a = 0, b = 1):
        self.range = b - a
        self.start = a

    def __call__(self, sample):
        noise = torch.rand_like(sample)
        return (noise * self.range + self.start) + sample

def readfile(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 128, 128, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :, :] = cv2.resize(img,(128, 128), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

def readfile_144(path, label):
    image_dir = sorted(os.listdir(path))
    x = np.zeros((len(image_dir), 144, 144, 3), dtype=np.uint8)
    y = np.zeros((len(image_dir)), dtype=np.uint8)
    for i, file in enumerate(image_dir):
        img = cv2.imread(os.path.join(path, file))
        x[i, :, :, :] = cv2.resize(img,(144, 144), interpolation=cv2.INTER_CUBIC)[:, :, ::-1]
        if label:
            y[i] = int(file.split("_")[0])
    if label:
        return x, y
    else:
        return x

if __name__ == '__main__':
    workspace_dir = './food-11'
    print("Reading data")
    train_x, train_y = readfile_144(os.path.join(workspace_dir, "training"), True)
    print("Size of training data = {}".format(len(train_x)))
    val_x, val_y = readfile_144(os.path.join(workspace_dir, "validation"), True)
    print("Size of validation data = {}".format(len(val_x)))
    test_x = readfile(os.path.join(workspace_dir, "testing"), False)
    print("Size of Testing data = {}".format(len(test_x)))
    pickle.dump((train_x, train_y, val_x, val_y, test_x), open('data_144.pkl', 'wb'))

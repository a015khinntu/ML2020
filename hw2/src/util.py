from torch.utils.data import Dataset
import torch.nn as nn


class DNNDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class TestDataset(Dataset):
    def __init__(self, x):
        self.x = x

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i]


class DNN(nn.Module):
    def __init__(self, input_size, hidden_dim=128, num_layers=3, dropout=0.1):
        super(DNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        linear_list = [[nn.Linear(
            in_features=hidden_dim,
            out_features=hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)]
            for _ in range(num_layers-2)]
        linear_list = sum(linear_list, [])
        linear_list = [nn.Linear(input_size, hidden_dim), nn.ReLU(), nn.BatchNorm1d(hidden_dim)] + linear_list + [nn.Linear(
            in_features=hidden_dim, out_features=1), nn.Sigmoid()]
        self.linear = nn.Sequential(*linear_list)

    def forward(self, x):
        return self.linear(x)

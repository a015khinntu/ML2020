import torch
import torch.nn as nn
from torch.utils.data import Dataset

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels // 4, 1, 1, 0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4,
                      kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1, 1, 0),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.cnn(x)

class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features):
        self.bn = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()
        self.lin = nn.Linear(in_features, out_features)
        nn.init.kaiming_normal_(self.lin.weight, 1)
        nn.init.constant_(self.lin.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        return self.lin(self.relu(self.bn(x)))

class ConvxBlock(nn.Module):
    '''
    gcd(in_channels, out_channels) % cardinality == 0
    '''

    def __init__(self, in_channels, out_channels, cardinality=4, kernel_size=3, stride=1, padding=1, dilation=1):
        super(ConvxBlock, self).__init__()
        self.cardinality = cardinality
        self.in_channels = in_channels
        self.out_channels = out_channels
        for i in range(self.cardinality):
            setattr(self, 'conv{}'.format(i), nn.Sequential(
                nn.BatchNorm2d(in_channels // self.cardinality),
                nn.ReLU(),
                nn.Conv2d(in_channels // self.cardinality, out_channels // self.cardinality,
                          kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
            ))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        splits = torch.split(x, self.in_channels // self.cardinality, dim=-3)
        chunks = []
        for i, split in enumerate(splits):
            c = getattr(self, 'conv{}'.format(i))(split)
            chunks.append(c)
        return torch.cat(chunks, dim=-3)

class DilationBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilationBlock, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels // 4, 1, 1, 0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, 1, 2, 2),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1, 1, 0),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.cnn(x)

class ConvStride(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvStride, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels // 4, 1, 1, 0),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, 2, 1),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1, 1, 0),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.cnn(x)

class FullConvStride(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FullConvStride, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 3, 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.cnn(x)

class ConvSeparable(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=32):
        super(ConvSeparable, self).__init__()
        self.cnn = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, hidden_channels, 3, 1),
            nn.Conv2d(hidden_channels, hidden_channels,
                      3, 1, groups=hidden_channels),
            nn.Conv2d(hidden_channels, out_channels, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.cnn(x)

class ConvMultiScale(nn.Module):
    '''
    out_channels is need to be divide by 4
    '''

    def __init__(self, in_channels, out_channels):
        super(ConvMultiScale, self).__init__()
        self.conv1 = ConvBlock(in_channels, out_channels // 4)
        self.conv2 = ConvBlock(
            in_channels, out_channels // 4, dilation=2, padding=2)
        self.conv3 = ConvBlock(
            in_channels, out_channels // 4, dilation=3, padding=3)
        self.conv4 = ConvBlock(
            in_channels, out_channels // 4, dilation=4, padding=4)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y3 = self.conv3(x)
        y4 = self.conv4(x)
        y = torch.cat((y1, y2, y3, y4), dim=-3)
        return y

class ConvHighway(nn.Module):
    def __init__(self, num_channels):
        super(ConvHighway, self).__init__()
        self.gate = nn.Sequential(
            ConvBlock(num_channels, num_channels),
            nn.Sigmoid()
        )
        self.transform = nn.Sequential(
            ConvBlock(num_channels, num_channels),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.transform(x)
        g = self.gate(x)
        return h * g + (1 - g) * x

class ResBlock(nn.Module):
    def __init__(self, num_channels):
        super(ResBlock, self).__init__()
        self.conv = ConvBlock(num_channels, num_channels)
        self.bn = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU()
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        y = self.conv(x)
        y = self.relu(self.bn(y + x))
        return y

class Classifier_v0(nn.Module):
    def __init__(self):
        super(Classifier_v0, self).__init__()
        #torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        #torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.conv = nn.Conv2d(3, 128, 5, 1, 1)    # 32, 128, 128
        self.res = nn.Sequential(
            ResBlock(128),                   # 32, 128, 128
            ConvStride(128, 256),                 # 64, 64, 64
            ResBlock(256),                  # 64, 64, 64
            ConvStride(256, 512),               # 128, 32, 32
            ResBlock(512),                 # 128, 32, 32
            ConvStride(512, 1024),               # 256, 16, 16
            ResBlock(1024),                 # 256, 16, 16
            nn.AvgPool2d(4, 4)          # 256, 4, 4
        )
        self.bn = nn.BatchNorm1d(1024 * 4 * 4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024 * 4 * 4, 11)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.res(x)
        x = x.view(batch_size, -1)
        x = self.fc(self.relu(self.bn(x)))
        return x

class Classifier_v1(nn.Module):
    def __init__(self):
        super(Classifier_v1, self).__init__()
        self.conv = nn.Conv2d(3, 64, 7, 1, 3)    # 32, 128, 128
        self.res = nn.Sequential(
            ResBlock(64),                       # 32, 128, 128
            ResBlock(64),                       # 32, 128, 128
            ResBlock(64),                       # 32, 128, 128
            ConvHighway(64),                    # 32, 128, 128
            ConvStride(64, 128),                # 64, 64, 64
            ResBlock(128),                      # 64, 64, 64
            ResBlock(128),                      # 64, 64, 64
            ResBlock(128),                      # 64, 64, 64
            ConvHighway(128),                   # 64, 64, 64
            ConvStride(128, 256),               # 128, 32, 32
            ResBlock(256),                      # 128, 32, 32
            ResBlock(256),                      # 128, 32, 32
            ResBlock(256),                      # 128, 32, 32
            ConvHighway(256),                   # 128, 32, 32
            ConvStride(256, 512),               # 256, 16, 16
            ResBlock(512),                      # 256, 16, 16
            ResBlock(512),                      # 256, 16, 16
            ResBlock(512),                      # 256, 16, 16
            ConvHighway(512),                   # 256, 16, 16
            nn.AvgPool2d(4, 4)                  # 256, 4, 4
        )
        self.bn = nn.BatchNorm1d(512 * 4 * 4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512 * 4 * 4, 11)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.res(x)
        x = x.view(batch_size, -1)
        x = self.fc(self.relu(self.bn(x)))
        return x

class Classifier_v2(nn.Module):
    def __init__(self):
        super(Classifier_v2, self).__init__()
        self.conv = nn.Conv2d(3, 64, 7, 1, 3)    # 32, 128, 128
        self.res_reuse1 = ResBlock(64)
        self.highway1 = ConvHighway(64)
        self.stride1 = ConvStride(64, 128)
        self.res_reuse2 = ResBlock(128)
        self.highway2 = ConvHighway(128)
        self.stride2 = ConvStride(128, 256)
        self.res_reuse3 = ResBlock(256)
        self.highway3 = ConvHighway(256)
        self.stride3 = ConvStride(256, 512)
        self.res_reuse4 = ResBlock(512)
        self.highway4 = ConvHighway(512)
        self.pool = nn.AvgPool2d(4, 4)
        self.bn = nn.BatchNorm1d(512 * 4 * 4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(512 * 4 * 4, 11)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.res_reuse1(self.res_reuse1(self.res_reuse1(x)))
        x = self.stride1(self.highway1(x))
        x = self.res_reuse2(self.res_reuse2(self.res_reuse2(x)))
        x = self.stride2(self.highway2(x))
        x = self.res_reuse3(self.res_reuse3(self.res_reuse3(x)))
        x = self.stride3(self.highway3(x))
        x = self.res_reuse4(self.res_reuse4(self.res_reuse4(x)))
        x = self.pool(self.highway4(x))
        x = x.view(batch_size, -1)
        x = self.fc(self.relu(self.bn(x)))
        return x

class Classifier_v3(nn.Module):
    '''
    resnet
    '''
    def __init__(self):
        super(Classifier_v3, self).__init__()
        self.conv = nn.Conv2d(3, 128, 7, 1, 3)      # 32, 128, 128
        self.res = nn.Sequential(
            ResBlock(128),                          # 32, 128, 128
            FullConvStride(128, 256),               # 64, 64, 64
            ResBlock(256),                          # 64, 64, 64
            FullConvStride(256, 512),               # 128, 32, 32
            ResBlock(512),                          # 128, 32, 32
            FullConvStride(512, 1024),              # 256, 16, 16
            ResBlock(1024),                         # 256, 16, 16
            nn.AvgPool2d(4, 4)                      # 256, 4, 4
        )
        self.bn = nn.BatchNorm1d(1024 * 4 * 4)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(1024 * 4 * 4, 11)
        nn.init.kaiming_normal_(self.conv.weight)
        nn.init.constant_(self.conv.bias, 0)
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)
        x = self.res(x)
        x = x.view(batch_size, -1)
        x = self.fc(self.relu(self.bn(x)))
        return x

class Classifier_v4(nn.Module):
    '''
    vgg-net with bottle neck, 13 conv layers
    '''
    def __init__(self):
        super(Classifier_v4, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 3),  # 64, 128, 128
            ConvBlock(128, 256),         # 128, 128, 128
            nn.MaxPool2d(2, 2),         # 128, 64, 64
            ConvBlock(256, 512),        # 256, 64, 64
            nn.MaxPool2d(2, 2),         # 256, 32, 32
            ConvBlock(512, 1024),        # 512, 32, 32
            nn.MaxPool2d(2, 2),         # 512, 16, 16
            ConvBlock(1024, 2048),       # 1024, 16, 16
            nn.MaxPool2d(2, 2)          # 1024, 8, 8
        )
        self.fc = nn.Linear(2048 * 8 * 8, 11)
        self.bn = nn.BatchNorm1d(2048 * 8 * 8)
        self.relu = nn.ReLU()
        nn.init.kaiming_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.cnn(x)
        x = x.view(batch_size, -1)
        x = self.fc(self.relu(self.bn(x)))
        return x

class Classifier_v5(nn.Module):
    '''
    self-attention on feature
    '''
    def __init__(self):
        super(Classifier_v5, self).__init__()
        self.res = nn.Sequential(
            nn.Conv2d(3, 128, 7, 1, 3),             # 128, 128, 128
            ResBlock(128),                          # 128, 128, 128
            FullConvStride(128, 256),               # 256, 64, 64
            ResBlock(256),                          # 256, 64, 64
            FullConvStride(256, 512),               # 512, 32, 32
            ResBlock(512),                          # 512, 32, 32
            FullConvStride(512, 1024),              # 1024, 16, 16
            ResBlock(1024),                         # 1024, 16, 16
            nn.MaxPool2d(2, 2)                      # 1024, 8, 8
        )
        self.fc = LinearBlock(1024 * 8 * 8, 11)
        self.gate = LinearBlock(1024, 1024)
        self.transform = LinearBlock(1024, 1024)
        self.act = nn.Sigmoid()
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = self.res(x)                                         # 1024, 4, 4
        x = x.permute(0, 2, 3, 1).view(batch_size, -1, 1024)    # 64, 1024
        h = self.transform(x)
        g = self.act(self.gate(x))
        x = h * g + (1 - g) * x
        x = self.fc(x)
        return x

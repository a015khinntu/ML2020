from nn_util import Classifier_v3 as Classifier
from torchsummary import summary
import torch.nn as nn
import torch
from time import time

t = time()

loss_fn = nn.CrossEntropyLoss()


model = Classifier()
summary(model, (3, 128, 128), 2, 'cpu')
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print('parameter total:{}, trainable:{}'.format(total, trainable))

# opt = torch.optim.Adam(model.parameters(), lr=1e-3)
# a = torch.randn(2, 3, 128, 128)
# b = model(a)
# target = torch.LongTensor([0, 1])
# # 
# loss = loss_fn(b, target)
# loss.backward()
# opt.step()
# print(time() - t)

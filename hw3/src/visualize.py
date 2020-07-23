from torchviz import make_dot
import torch
from util import ConvBlock

model = ConvBlock(32, 32)

x = torch.randn(2, 32, 128, 128)

make_dot(model(x), params=dict(model.named_parameters())).render("attached", format="png")

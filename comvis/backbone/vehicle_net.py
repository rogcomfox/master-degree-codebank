# Based on Paper "Vehicle classification using a real-time convolutional structure based on DWT pooling layer and SE blocks"
import torch.nn as nn
import torch.nn.functional as F
import pywt
import math
import numpy as np
import torch
from torchsummary import summary

class HSigmoid(nn.Module):
    """
    Approximated sigmoid function, so-called hard-version of sigmoid from 'Searching for MobileNetV3,'
    https://arxiv.org/abs/1905.02244.
    """
    def forward(self, x):
        return F.relu6(x + 3.0, inplace=True) / 6.0
    
class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block from 'Squeeze-and-Excitation Networks,' https://arxiv.org/abs/1709.01507.
    Parameters:
    ----------
    channels : int, Number of channels.
    reduction : int, default 16 queeze reduction value.
    approx_sigmoid : bool, default False Whether to use approximated sigmoid function.
    activation : function, or str, or nn.Module Activation function or name of activation function.
    """
    def __init__(self, channels, reduction=16, approx_sigmoid=True):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

class PoolingLayer(nn.Module):
    def __init__(self, n_features):
        super(PoolingLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(num_features=n_features, affine=True),
            nn.ReLU(inplace=False),
            SEBlock(n_features)
        )
    def forward(self, x):
        return self.layer(x)

class VehicleNet(nn.Module):
    def __init__(self):
        super(VehicleNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1),
            PoolingLayer(32)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1),
            PoolingLayer(32)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1),
            PoolingLayer(64)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=5, stride=1),
            PoolingLayer(128)
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=5, stride=1),
            PoolingLayer(256)
        )
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(32400, 256)
        self.fc2 = nn.Linear(256, 17)
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.dropout(out)
        out = torch.flatten(out, 2)
        out = self.fc1(out)
        out = self.fc2(out)
        return F.log_softmax(out, 1)

model = VehicleNet()
print(summary(model.cuda(), input_size=(3, 200, 200)))

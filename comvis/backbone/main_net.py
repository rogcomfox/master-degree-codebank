# building the cnn network
import torch
import torch.nn as nn
import torch.nn.functional as F
import backbone.pooling_layer as pool

class MainNet(nn.Module):
    def __init__(self):
        super(MainNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=5, stride=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=5, stride=1)
        self.pool1 = pool.PoolingLayer(32)
        self.pool2 = pool.PoolingLayer(32)
        self.pool3 = pool.PoolingLayer(64)
        self.pool4 = pool.PoolingLayer(128)
        self.pool5 = pool.PoolingLayer(256)
        self.fc1 = nn.Linear(16384, 256)
        self.fc2 = nn.Linear(256, 5)
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool5(F.relu(self.conv5(x)))
        x = torch.dropout(x, 0.2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
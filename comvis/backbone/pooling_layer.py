import torch.nn as nn
import torch.nn.functional as F
import backbone.dwt_layer as dwt
import backbone.se_layer as se

# pooling layer recipe consist DWT + BN + ReLu + SE Layer
class PoolingLayer(nn.Module):
    def __init__(self, n_features) -> None:
        super(PoolingLayer, self).__init__()
        self.dwt = dwt.HaarForward()
        self.indwt = dwt.HaarInverse()
        self.bn = nn.BatchNorm2d(n_features)
        self.se = se.SE(n_features)
    
    def forward(self, x):
        x = self.dwt(x)
        x = self.indwt(x)
        x = self.bn(x)
        x = F.relu(x)
        output = self.se(x)
        return output
#MCNN based on paper (Image Classification Using Multiple Convolutional Neural Networks on the Fashion-MNIST Dataset)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

class MCNN9(nn.Module):
    def __init__(self):
        super(MCNN9, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 256, kernel_size=3),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.Conv2d(256, 192, kernel_size=3)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(192, 256, kernel_size=3),
            nn.Conv2d(256, 32, kernel_size=3),
            nn.Conv2d(32, 192, kernel_size=3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(192, 192, kernel_size=3),
            nn.Conv2d(192, 128, kernel_size=3),
            nn.Conv2d(128, 32, kernel_size=3)
        )
        self.pool = nn.Sequential(nn.MaxPool2d(2,2))
        self.fc = nn.Linear(15488, 64)

    def forward(self,x):
        out = self.block1(x)
        out = self.pool(out)
        out = self.block2(out)
        out = self.pool(out)
        out = self.block3(out)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=1)
        # out = torch.transpose(out, 0, 1)
        out = self.fc(out)
        return F.log_softmax(out, 1)

class MCNN12(nn.Module):
    def __init__(self):
        super(MCNN12, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.Conv2d(32, 128, kernel_size=3),
            nn.Conv2d(128, 32, kernel_size=3),
            nn.Conv2d(32, 192, kernel_size=3)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(192, 64, kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Conv2d(128, 32, kernel_size=3),
            nn.Conv2d(32, 256, kernel_size=3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(256, 192, kernel_size=3),
            nn.Conv2d(192, 32, kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Conv2d(64, 32, kernel_size=3)
        )
        self.pool = nn.Sequential(nn.MaxPool2d(2,2))
        self.fc = nn.Linear(14112, 32)

    def forward(self,x):
        out = self.block1(x)
        out = self.pool(out)
        out = self.block2(out)
        out = self.pool(out)
        out = self.block3(out)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=1)
        # out = torch.transpose(out, 0, 1)
        out = self.fc(out)
        return F.log_softmax(out, 1)

# class MCNN15(nn.Module):
#     def __init__(self):
#         super(MCNN9, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 256, kernel_size=3),
#             nn.Conv2d(256, 256, kernel_size=3),
#             nn.Conv2d(256, 192, kernel_size=3)
#         )
#         self.block2 = nn.Sequential(
#             nn.Conv2d(192, 256, kernel_size=3),
#             nn.Conv2d(256, 32, kernel_size=3),
#             nn.Conv2d(32, 192, kernel_size=3)
#         )
#         self.block3 = nn.Sequential(
#             nn.Conv2d(192, 192, kernel_size=3),
#             nn.Conv2d(192, 128, kernel_size=3),
#             nn.Conv2d(128, 32, kernel_size=3)
#         )
#         self.pool = nn.Sequential(nn.MaxPool2d(2,2))
#         self.fc = nn.Linear(15488, 64)

#     def forward(self,x):
#         out = self.block1(x)
#         out = self.pool(out)
#         out = self.block2(out)
#         out = self.pool(out)
#         out = self.block3(out)
#         out = self.pool(out)
#         out = torch.flatten(out, start_dim=1)
#         # out = torch.transpose(out, 0, 1)
#         out = self.fc(out)
#         return F.log_softmax(out, 1)

class MCNN18(nn.Module):
    def __init__(self):
        super(MCNN18, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3),
            nn.Conv2d(128, 32, kernel_size=3),
            nn.Conv2d(32, 32, kernel_size=3),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3)
        )
        self.pool = nn.Sequential(nn.MaxPool2d(2,2))
        self.block2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3),
            nn.Conv2d(256, 192, kernel_size=3),
            nn.Conv2d(192, 256, kernel_size=3),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.Conv2d(128, 64, kernel_size=3)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.Conv2d(64, 256, kernel_size=3),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.Conv2d(256, 32, kernel_size=3),
            nn.Conv2d(32, 256, kernel_size=3),
            nn.Conv2d(256, 32, kernel_size=3)
        )
        self.fc = nn.Linear(9248, 64)
    
    def forward(self,x):
        out = self.block1(x)
        out = self.pool(out)
        out = self.block2(out)
        out = self.pool(out)
        out = self.block3(out)
        out = self.pool(out)
        out = torch.flatten(out, start_dim=1)
        # out = torch.transpose(out, 0, 1)
        out = self.fc(out)
        return F.log_softmax(out, 1)

model = MCNN18()
print(summary(model.cuda(), input_size=(3, 224, 224)))

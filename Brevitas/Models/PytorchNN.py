"""
Created on Tue May  4 12:45:58 2021

Basic Model is creted with the help of Pytorch.
Results from this Model can be taken as a baseline for Quantization Aware Training

@author: uslumt
"""

import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.BN1 = nn.BatchNorm2d(6, affine = False)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.BN2 = nn.BatchNorm2d(16, affine = False)
        self.fc1 = nn.Linear(400, 200)
        self.fc2 = nn.Linear(200, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = self.BN1(self.pool(self.conv1(x)))
        x = F.relu(x)
        x = self.BN2(self.pool(self.conv2(x)))
        x = F.relu(x)
        x = x.view(-1, 400)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

"""
Defines the convolutional neural network with 3 convolutional and 3 fully-connected layers.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import numpy as np

class Net3(nn.Module):

    def __init__(self):
        super(Net3, self).__init__()
        # 3 input channels, 6 output channels, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5, stride=2)
        self.conv2 = nn.Conv2d(6, 16, 5, stride=2)
        self.conv3 = nn.Conv2d(16, 32, 3, stride=2)

        # an affine operation: y = Wx + b
        # need to adjust the first number here to whatever one gets in
        self.fc1 = nn.Linear(6272, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

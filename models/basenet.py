"""
Base net for experiments. Consists of a few convolutions and activations
"""

import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):

    def __init__(self, in_ch, num_classes):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

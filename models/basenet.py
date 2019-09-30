"""
Base net for experiments. Consists of a few convolutions and activations
"""

import torch.nn as nn
import torch.nn.functional as F


class BaseNet(nn.Module):

    def __init__(self, in_ch, num_classes, act_fact):
        super(BaseNet, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(5*5*50, 500)
        self.fc2 = nn.Linear(500, num_classes)
        self.act1 = act_fact()
        self.act2 = act_fact()
        self.act3 = act_fact()

    def forward(self, x):
        x = self.act1(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.act2(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 5*5*50)
        x = self.act3(self.fc1(x))
        x = self.fc2(x)
        return x

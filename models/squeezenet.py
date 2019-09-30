"""
squeezenet in pytorch

Inspired from: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/

[1] Song Han, Jeff Pool, John Tran, William J. Dally
    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

import torch
import torch.nn as nn


class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):
        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.BatchNorm2d(squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.BatchNorm2d(int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x


class SqueezeNet(nn.Module):
    """mobile net with simple bypass"""

    def __init__(self, in_ch, num_classes):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(24, 32, 8)
        self.fire3 = Fire(32, 32, 8)
        self.fire4 = Fire(32, 64, 16)
        self.fire5 = Fire(64, 64, 16)
        self.fire6 = Fire(64, 96, 24)
        self.fire7 = Fire(96, 96, 24)
        self.fire8 = Fire(96, 128, 32)
        self.fire9 = Fire(128, 128, 32)

        self.conv10 = nn.Conv2d(128, num_classes, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        x = self.stem(x)

        f2 = self.fire2(x)
        f3 = self.fire3(f2) + f2
        f4 = self.fire4(f3)
        f4 = self.maxpool(f4)

        f5 = self.fire5(f4) + f4
        f6 = self.fire6(f5)
        f7 = self.fire7(f6) + f6
        f8 = self.fire8(f7)
        f8 = self.maxpool(f8)

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)

        return x


def squeezenet(in_ch, num_classes):
    return SqueezeNet(in_ch, num_classes)
"""
Resnet implementation in pytorch

Inspired from: https://github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.

    Deep Residual Learning for Image Recognition
    https://arxiv.org/abs/1512.03385v1
"""

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    """
    Basic Block for resnet 18 and resnet 34
    """

    expansion = 1

    def __init__(self, in_channels, out_channels, act_fact, stride=1):
        super().__init__()

        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            act_fact(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # shortcut
        self.shortcut = nn.Sequential()
        self.activation = act_fact()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        return self.activation(self.residual_function(x) + self.shortcut(x))


class ResNet(nn.Module):

    def __init__(self, block, num_block, in_ch, num_classes, act_fact):
        super().__init__()

        self.in_channels = 16

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            act_fact())
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 16, num_block[0], act_fact, 1)
        self.conv3_x = self._make_layer(block, 32, num_block[1], act_fact, 2)
        self.conv4_x = self._make_layer(block, 64, num_block[2], act_fact, 2)
        self.conv5_x = self._make_layer(block, 128, num_block[3], act_fact, 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, act_fact, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, act_fact, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)

        return output


def ResNet18(in_ch, num_classes, act_fact):
    """ return a ResNet 18 object
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], in_ch, num_classes, act_fact)


def ResNet34(in_ch, num_classes, act_fact):
    """ return a ResNet 34 object
    """
    return ResNet(BasicBlock, [3, 4, 6, 3], in_ch, num_classes, act_fact)

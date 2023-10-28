from random import shuffle
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import time
import torch.utils.data as Data
import os
import pandas as pd
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import datasets, models, transforms
cudnn.benchmark = True


class Residual(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.max = nn.MaxPool2d(2, 2)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        residual = self.conv3(residual)
        # print("residual . shape::::", residual.shape)
        # print("x . shape :::::::", x.shape)
        out = x + residual
        return self.max(out)


def resnet_block(in_channels, out_channels):
    blk = []
    blk.append(Residual(in_channels, out_channels))
    return nn.Sequential(*blk)


# # 需要注意的SE模块不是传统意义上的注意力机制，但是它仍然可以看作是一
# # 种轻量级的注意力机制，因为它也能够自适应地调整每个通道的权重。
# # 因此，SE模块也被称为通道注意力机制（channel attention mechanism）。
# class SEBlock(nn.Module):
#     def __init__(self, in_channels, reduction=16):
#         super(SEBlock, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc1 = nn.Linear(in_channels, in_channels // reduction)
#         self.fc2 = nn.Linear(in_channels // reduction, in_channels)
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         batch_size, channels, _, _ = x.size()
#         y = self.avg_pool(x).view(batch_size, channels)
#         y = self.fc1(y)
#         y = self.relu(y)
#         y = self.fc2(y)
#         y = self.sigmoid(y).view(batch_size, channels, 1, 1)
#         return x * y
#
#
# class Residual(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(Residual, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.bn3 = nn.BatchNorm2d(out_channels)
#         self.max = nn.MaxPool2d(2, 2)
#         self.se = SEBlock(out_channels)
#
#     def forward(self, x):
#         residual = x
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = F.relu(self.bn2(self.conv2(x)))
#         residual = self.conv3(residual)
#         residual = self.bn3(residual)
#         x = self.se(x) + residual
#         out = F.relu(x)
#         return self.max(out)
#
#
# def resnet_block(in_channels, out_channels):
#     blk = []
#     blk.append(Residual(in_channels, out_channels))
#     return nn.Sequential(*blk)

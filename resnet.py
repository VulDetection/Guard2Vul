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
# train_transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor()
# ])
#
# val_transform = transforms.Compose([
#     transforms.ToTensor()
# ])

# batch_size = 64

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
#                                         download=True, transform=train_transform)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
#                                           shuffle=True, num_workers=2)
#
# testset = torchvision.datasets.CIFAR10(root='./data', train=False,
#                                        download=True, transform=val_transform)
# testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
#                                          shuffle=False, num_workers=2)


#
# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(f"Using {device} device")


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


# net = nn.Sequential()
# net.add_module("resnet_block1", resnet_block(3, 128))
# net.add_module("resnet_block2", resnet_block(128, 256))
# net.add_module("resnet_block3", resnet_block(256, 512))
# fc = nn.Sequential(
#     nn.Conv2d(512, 1024, kernel_size=3, padding=1),
#     nn.BatchNorm2d(1024),
#     nn.ReLU(),
#     nn.AvgPool2d(4, 4),
#     nn.Flatten(),
#     nn.Dropout(0.5),
#     nn.Linear(1024, 10),
#
# )
#
# net.add_module("fc", fc)
# model = net.to(device)
# print(model)
# model


# def train(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     print(num_batches)
#     model.train()
#     train_loss, correct = 0, 0
#     for batch, (X, y) in enumerate(dataloader):
#         X, y = X.to(device), y.to(device)
#
#         # Compute prediction error
#         pred = model(X)
#         loss = loss_fn(pred, y)
#
#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss_fn(pred, y).item()
#         correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#         if batch % 100 == 0:
#             loss, current = loss.item(), batch * len(X)
#             # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
#     train_loss /= num_batches
#     correct /= size
#
#     print(f"train Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {train_loss:>8f} \n")
#
#
# def val(dataloader, model, loss_fn):
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     model.eval()
#     val_loss, correct = 0, 0
#     with torch.no_grad():
#         for X, y in dataloader:
#             X, y = X.to(device), y.to(device)
#             pred = model(X)
#             val_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()
#     val_loss /= num_batches
#     correct /= size
#     print(f"val Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {val_loss:>8f} \n")
#
#
# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# epochs = 50
#
# since = time.time()
# for t in range(epochs):
#     print(f"Epoch {t + 1}\n-------------------------------")
#     train(trainloader, model, loss_fn, optimizer)
#     val(testloader, model, loss_fn)
# time_elapsed = time.time() - since
# print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
# print("Done!")
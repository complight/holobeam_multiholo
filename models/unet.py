""" Full assembly of the parts to form the complete network """
import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from .unet_parts import *

import math

class Res_Block(nn.Module):
    def __init__(self, dims=32):
        super(Res_Block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.instance_norm = nn.InstanceNorm2d(dims)
        self.conv2 = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        res = self.conv1(x)
        res = self.instance_norm(res)
        res = self.relu(res)
        res = self.conv2(res)
        return x + res

## Revised from https://github.com/milesial/Pytorch-UNet/tree/master/unet
class UNet(nn.Module):
    def __init__(self, dims=64, in_channels=2, out_channels=1, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(in_channels, dims)
        self.down1 = Down(dims, dims*2)
        self.down2 = Down(dims*2, dims*4)
        self.down3 = Down(dims*4, dims*8)
        factor = 2 if bilinear else 1
        self.down4 = Down(dims*8, dims*16 // factor)

        self.up1 = Up(dims*16, dims*8 // factor, bilinear)
        self.up2 = Up(dims*8, dims*4 // factor, bilinear)
        self.up3 = Up(dims*4, dims*2 // factor, bilinear)
        self.up4 = Up(dims*2, dims, bilinear)
        self.outc = OutConv(dims, out_channels)

    def make_layer(self, block, num_of_layer, dims):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block(dims))
        return nn.Sequential(*layers)


    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out

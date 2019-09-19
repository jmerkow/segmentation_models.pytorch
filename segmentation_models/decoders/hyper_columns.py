import torch
from torch import nn as nn
from torch.nn import functional as F


class HyperColumnBlock(nn.Module):

    def __init__(self, decoder_channels, final_channels):
        super().__init__()
        self.layer1_conv = nn.Conv2d(decoder_channels[0], final_channels, kernel_size=(1, 1))
        self.layer2_conv = nn.Conv2d(decoder_channels[1], final_channels, kernel_size=(1, 1))
        self.layer3_conv = nn.Conv2d(decoder_channels[2], final_channels, kernel_size=(1, 1))
        self.layer4_conv = nn.Conv2d(decoder_channels[3], final_channels, kernel_size=(1, 1))
        self.layer5_conv = nn.Conv2d(decoder_channels[4], final_channels, kernel_size=(1, 1))
        self.final_conv = nn.Conv2d(5, final_channels, kernel_size=(1, 1))

    def forward(self, x):
        x1 = self.layer1_conv(x[0])
        x2 = self.layer2_conv(x[1])
        x3 = self.layer3_conv(x[2])
        x4 = self.layer4_conv(x[3])
        x5 = self.layer5_conv(x[4])

        x = torch.cat([
            F.interpolate(x1, scale_factor=16, mode='nearest'),
            F.interpolate(x2, scale_factor=8, mode='nearest'),
            F.interpolate(x3, scale_factor=4, mode='nearest'),
            F.interpolate(x4, scale_factor=2, mode='nearest'),
            x5
        ], dim=1)

        return self.final_conv(x)

# -*- coding: utf-8 -*-

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from segmentation_models.base.model import Model
from segmentation_models.common.blocks import Conv2dReLU
from .hyper_columns import HyperColumnBlock
from .unet import CenterBlock


class UNetPPDecoder(Model):

    def compute_channels(self, en, dc):

        en5 = 0  # just to make it pretty
        channels = [
            [en[1] + en[0]],
            [en[2] + en[1], en[2] + dc[0] + dc[1]],
            [en[3] + en[2], en[3] + dc[2] + dc[1], en[3] + dc[2] + dc[1]],
            [en[4] + en[3], en[4] + dc[3] + dc[2], en[4] + dc[3] + dc[2], en[4] + dc[3] + dc[2]],
            [en5 + en[4], en5 + dc[4] + dc[3], en5 + dc[4] + dc[3], en5 + dc[4] + dc[3], en5 + dc[4] + dc[3]]
        ]

        return channels

    def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16),
                 final_channels=1,
                 use_batchnorm=True,
                 dropout=0,
                 upsample_mode='bilinear',
                 hyper_columns=False,
                 center=True,
                 **conv_params):
        super().__init__()

        if center:
            channels = encoder_channels[0]
            self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
        else:
            self.center = None

        channels = self.compute_channels(encoder_channels, decoder_channels)
        if dropout:
            self.dropout = dropout
        else:
            self.dropout = None

        if upsample_mode == 'bilinear':
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif upsample_mode == 'nearest':
            self.up = nn.Upsample(scale_factor=2, mode='nearest')

        block_params = dict(kernel_size=3, padding=1, use_batchnorm=use_batchnorm, **conv_params)
        self.block41 = Conv2dReLU(channels[0][0], decoder_channels[0], **block_params)

        self.block31 = Conv2dReLU(channels[1][0], decoder_channels[1], **block_params)
        self.block32 = Conv2dReLU(channels[1][1], decoder_channels[1], **block_params)

        self.block21 = Conv2dReLU(channels[2][0], decoder_channels[2], **block_params)
        self.block22 = Conv2dReLU(channels[2][1], decoder_channels[2], **block_params)
        self.block23 = Conv2dReLU(channels[2][2], decoder_channels[2], **block_params)

        self.block11 = Conv2dReLU(channels[3][0], decoder_channels[3], **block_params)
        self.block12 = Conv2dReLU(channels[3][1], decoder_channels[3], **block_params)
        self.block13 = Conv2dReLU(channels[3][2], decoder_channels[3], **block_params)
        self.block14 = Conv2dReLU(channels[3][3], decoder_channels[3], **block_params)

        self.block01 = Conv2dReLU(channels[4][0], decoder_channels[4], **block_params)
        self.block02 = Conv2dReLU(channels[4][1], decoder_channels[4], **block_params)
        self.block03 = Conv2dReLU(channels[4][2], decoder_channels[4], **block_params)
        self.block04 = Conv2dReLU(channels[4][3], decoder_channels[4], **block_params)
        self.block05 = Conv2dReLU(channels[4][4], decoder_channels[4], **block_params)

        if hyper_columns:
            self.hyper_columns = HyperColumnBlock(decoder_channels, final_channels)
        else:
            self.hyper_columns = None
            self.final_conv = nn.Conv2d(decoder_channels[4], final_channels, kernel_size=(1, 1))

        self.initialize()

    def forward(self, x):

        cat_normal = partial(torch.cat, dim=1)
        if self.dropout:
            cat_dropout = lambda y: F.dropout(cat_normal(y), self.dropout, self.training, inplace=True)
            cat = cat_dropout
        else:
            cat = cat_normal

        up = self.up

        encoder_head = x[0]

        if self.center:
            encoder_head = self.center(encoder_head)

        x41 = self.block41(cat([up(encoder_head), x[1]]))

        x31 = self.block31(cat([up(x[1]), x[2]]))
        x32 = self.block32(cat([up(x41), x31, x[2]]))

        x21 = self.block21(cat([up(x[2]), x[3]]))
        x22 = self.block22(cat([up(x31), x21, x[3]]))
        x23 = self.block23(cat([up(x32), x22, x[3]]))

        x11 = self.block11(cat([up(x[3]), x[4]]))
        x12 = self.block12(cat([up(x21), x11, x[4]]))
        x13 = self.block13(cat([up(x22), x12, x[4]]))
        x14 = self.block14(cat([up(x23), x13, x[4]]))

        x01 = self.block01(up(x[4]))
        x02 = self.block02(cat([up(x11), x01]))
        x03 = self.block03(cat([up(x12), x02]))
        x04 = self.block04(cat([up(x13), x03]))
        x05 = self.block05(cat([up(x14), x04]))

        if self.hyper_columns:
            x = self.hyper_columns([x41, x32, x23, x14, x05])
        else:
            x = self.final_conv(x05)
        return x

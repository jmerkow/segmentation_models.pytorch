import torch
import torch.nn as nn
import torch.nn.functional as F


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                     stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)


class ASPP(nn.Module):
    def __init__(self, inplanes, output_stride, mask_channels=256):
        super(ASPP, self).__init__()
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, mask_channels, 1, padding=0, dilation=dilations[0])
        self.aspp2 = _ASPPModule(inplanes, mask_channels, 3, padding=dilations[1], dilation=dilations[1])
        self.aspp3 = _ASPPModule(inplanes, mask_channels, 3, padding=dilations[2], dilation=dilations[2])
        self.aspp4 = _ASPPModule(inplanes, mask_channels, 3, padding=dilations[3], dilation=dilations[3])

        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, mask_channels, 1, stride=1, bias=False),
                                             nn.BatchNorm2d(mask_channels),
                                             nn.ReLU())
        self.conv1 = nn.Conv2d(mask_channels * 5, mask_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mask_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)


class Decoder(nn.Module):
    def __init__(self, num_classes, low_level_inplanes, decoder_channels=(256, 48)):
        super(Decoder, self).__init__()

        self.conv1 = nn.Conv2d(low_level_inplanes, decoder_channels[1], 1, bias=False)
        self.bn1 = nn.BatchNorm2d(decoder_channels[1])
        self.relu = nn.ReLU()
        self.last_conv = nn.Sequential(nn.Conv2d(sum(decoder_channels),
                                                 decoder_channels[0], kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(decoder_channels[0]),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(decoder_channels[0], decoder_channels[0], kernel_size=3, stride=1,
                                                 padding=1, bias=False),
                                       nn.BatchNorm2d(decoder_channels[0]),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(decoder_channels[0], num_classes, kernel_size=1, stride=1))

    def forward(self, mask, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        x = F.interpolate(mask, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x


class DeepLabDecoder(nn.Module):

    def __init__(self, encoder_channels, decoder_channels=(256, 48), final_channels=1, output_stride=8):
        super(DeepLabDecoder, self).__init__()
        self.aspp = ASPP(encoder_channels[0], output_stride=output_stride, mask_channels=decoder_channels[0])
        self.decoder = Decoder(final_channels, encoder_channels[3], decoder_channels=decoder_channels)

    def forward(self, x):
        features = x[0]
        low_level_features = x[3]

        mask = self.aspp(features)
        mask = self.decoder(mask, low_level_features)
        mask = F.interpolate(mask, scale_factor=4, mode='bilinear', align_corners=True)

        return mask

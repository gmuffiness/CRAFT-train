"""  
Copyright (c) 2019-present NAVER Corp.
MIT License
"""

# -*- coding: utf-8 -*-
import torchvision
from torch import nn
import torch
import torch.nn.functional as F

class ConvBlock(nn.Module):

    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_non_linearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU()
        self.with_non_linearity = with_non_linearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_non_linearity:
            x = self.ReLU(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="bilinear_craft"):
        super().__init__()

        if up_conv_in_channels is None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels is None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        elif upsampling_method == "bilinear_craft":
             self.upsample = "bilinear_craft"


        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

        self.conv_adj = nn.Conv2d(up_conv_in_channels, up_conv_out_channels, kernel_size=1, stride=1)

    def craft_upsample(self, up_x, down_x):
        up_x = F.interpolate(up_x, size=down_x.size()[2:], mode='bilinear', align_corners=False)
        up_x = self.conv_adj(up_x)
        return up_x

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: up-sampled feature map
        """
        if up_x.shape[2] != down_x.shape[2] :
            if self.upsample == "bilinear_craft":
                up_x = self.craft_upsample(up_x, down_x)
            else:
                up_x = self.upsample(up_x)

        x = torch.cat([up_x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x



class UNetWithResnet50Encoder(nn.Module):
    DEPTH = 6

    def __init__(self, pretrained=False, n_classes=2, amp=False):
        super().__init__()
        resnet = torchvision.models.resnet.resnet50(pretrained=pretrained)
        down_blocks = []
        up_blocks = []
        self.amp = amp
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(2048, 2048)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2), stride=2)

        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=4096, out_channels=2048,
                                                    up_conv_in_channels=2048, up_conv_out_channels=2048))
        up_blocks.append(UpBlockForUNetWithResNet50(2048, 1024))
        up_blocks.append(UpBlockForUNetWithResNet50(1024, 512))
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=128 + 64, out_channels=128,
                                                    up_conv_in_channels=256, up_conv_out_channels=128))
        # up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 3, out_channels=64,
        #                                             up_conv_in_channels=128, up_conv_out_channels=64))

        self.up_blocks = nn.ModuleList(up_blocks)

        self.conv_cls = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, n_classes, kernel_size=1),
        )

        self.final_out = nn.Sigmoid()

    def forward(self, x, with_output_feature_map=True):
        if self.amp:
            with torch.cuda.amp.autocast():
                pre_pools = dict()
                # pre_pools[f"layer_0"] = x
                x = self.input_block(x)
                pre_pools[f"layer_0"] = x
                x = self.input_pool(x)

                for i, block in enumerate(self.down_blocks, 1):
                    x = block(x)
                    if i == (UNetWithResnet50Encoder.DEPTH - 1):
                        continue
                    pre_pools[f"layer_{i}"] = x

                x = self.maxPool(x)
                x = self.bridge(x)

                for i, block in enumerate(self.up_blocks, 1):
                    key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
                    x = block(x, pre_pools[key])

                output_feature_map = x
                x = self.conv_cls(x)
                x = self.final_out(x)

                del pre_pools
                if with_output_feature_map:
                    return x.permute(0, 2, 3, 1), output_feature_map
                else:
                    return x
        else:
            pre_pools = dict()
            # pre_pools[f"layer_0"] = x
            x = self.input_block(x)
            pre_pools[f"layer_0"] = x
            x = self.input_pool(x)

            for i, block in enumerate(self.down_blocks, 1):
                x = block(x)
                if i == (UNetWithResnet50Encoder.DEPTH - 1):
                    continue
                pre_pools[f"layer_{i}"] = x

            x = self.maxPool(x)
            x = self.bridge(x)

            for i, block in enumerate(self.up_blocks, 1):
                key = f"layer_{UNetWithResnet50Encoder.DEPTH - 1 - i}"
                x = block(x, pre_pools[key])

            output_feature_map = x
            x = self.conv_cls(x)
            x = self.final_out(x)

            del pre_pools
            if with_output_feature_map:
                return x.permute(0, 2, 3, 1), output_feature_map
            else:
                return x


if __name__ == '__main__':
    from torchsummary import summary

    model = UNetWithResnet50Encoder(pretrained=True, n_classes=2).cuda()
    summary(model, input_size=(3, 768, 765))
    #output = model(torch.randn(1, 3, 768, 768).cuda())
    #print(output.shape)
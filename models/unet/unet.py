""" U-Net """
from __future__ import print_function
import math
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable

# four submodules: DoubleConv, Up, Down, OutConv
class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, inplace=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=inplace),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=inplace)
        )

    def forward(self, x):
        return self.double_conv(x)

class DoubleConv3(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, inplace=False):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, inplace=inplace)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Down3(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        self.up = nn.ConvTranspose2d (in_channels, in_channels // 2, kernel_size=3, stride=2, padding=1)
        self.conv = DoubleConv (out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size ()[2] - x1.size ()[2]
        diffX = x2.size ()[3] - x1.size ()[3]

        x1 = F.pad (x1, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class Up3(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, mid_channels=None, bilinear=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv3(in_channels, out_channels, in_channels // 2)
        else:
            if mid_channels is None:
                mid_channels = in_channels
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3(mid_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHWZ
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        # diffZ = x2.size()[4] - x1.size()[4]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2,
        #                 diffZ // 2, diffY - diffZ // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class OutConv3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=1)

    def forward(self, x):
        return self.conv(x)


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            # nn.GELU(),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            # nn.GELU(),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
            # nn.GELU()
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):

            if i == 0:
                x1 = self.conv(x)

            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class single_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU (inplace=True)
        # self.relu = nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class UNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, bilinear=True):
        super(UNet, self).__init__()

        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, out_ch)

    def forward(self, x):
        x1 = self.inc(x)      # ------- 1
        x2 = self.down1(x1)   # ------- 2
        x3 = self.down2(x2)   # ------- 3
        x4 = self.down3(x3)   # ------- 4
        x5 = self.down4(x4)   # ------- 5
        x = self.up1(x5, x4)  # ------- 4' upsample x5 + concat and conv. them all.
        x = self.up2(x, x3)   # ------- 3'
        x = self.up3(x, x2)   # ------- 2'
        x = self.up4(x, x1)   # ------- 1'
        logits = self.outc(x)
        return logits


class ATUNet_Encoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(ATUNet_Encoder, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class ATUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, loader_conf=None, split_last=True):
        super(ATUNet, self).__init__()
        self.loader_conf = loader_conf
        self.split_last = split_last
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=256)
        self.Att3 = Attention_block(F_g=256, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=384, ch_out=256)

        self.Up2 = up_conv(ch_in=256, ch_out=128)
        self.Att2 = Attention_block(F_g=128, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=192, ch_out=128)
        self.conv6 = conv_block(ch_in=128, ch_out=128)

        # previous settings.
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.conv6 = conv_block(ch_in=64, ch_out=64)

        if self.split_last:
            self.Conv_1x1_f = nn.Conv2d(32, out_ch//2, kernel_size=1, stride=1, padding=0)
            self.Conv_1x1_b = nn.Conv2d(32, out_ch//2, kernel_size=1, stride=1, padding=0)
        else:
            self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.conv6(d2)

        if self.split_last:
            z1, z2 = torch.chunk(d2, chunks=2, dim=1)
            z1_2 = self.Conv_1x1_f(z1)
            z2_2 = self.Conv_1x1_b(z2)
            d1 = torch.cat((z1_2, z2_2), dim=1)
        else:
            d1 = self.Conv_1x1(d2)
        return d1


class ATUNet_UV(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, loader_conf=None, split_last=True):
        super(ATUNet_UV, self).__init__()
        self.loader_conf = loader_conf
        self.split_last = split_last
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=256)
        self.Att3 = Attention_block(F_g=256, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=384, ch_out=256)

        self.Up2 = up_conv(ch_in=256, ch_out=128)
        self.Att2 = Attention_block(F_g=128, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=192, ch_out=128)

        # previous settings.
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.conv6 = conv_block(ch_in=64, ch_out=64)

        self.mask_act = nn.ReLU(inplace=True)

        if self.split_last:
            self.Conv_1x1_f = nn.Conv2d(32, out_ch // 2, kernel_size=1, stride=1, padding=0)
            self.Conv_1x1_b = nn.Conv2d(32, out_ch // 2, kernel_size=1, stride=1, padding=0)
        else:
            self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.conv6(d2)

        if self.split_last:
            z1, z2 = torch.chunk(d2, chunks=2, dim=1)
            z1_2 = self.Conv_1x1_f(z1)
            z2_2 = self.Conv_1x1_b(z2)
            d1 = torch.cat((z1_2, z2_2), dim=1)
        else:
            d1 = self.Conv_1x1(d2)
        return d1, d2

# class ATUNet_Color(nn.Module):
#     def __init__(self, in_ch=3, out_ch=1, loader_conf=None, split_last=True):
#         super(ATUNet_Color, self).__init__()
#         self.loader_conf = loader_conf
#         self.split_last = split_last
#         self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
#
#         self.Conv1 = conv_block(ch_in=in_ch, ch_out=64)
#         self.Conv2 = conv_block(ch_in=64, ch_out=128)
#         self.Conv3 = conv_block(ch_in=128, ch_out=256)
#         self.Conv4 = conv_block(ch_in=256, ch_out=512)
#         # self.Conv5 = conv_block(ch_in=512, ch_out=1024)
#
#         # self.Up5 = up_conv(ch_in=1024, ch_out=512)
#         # self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
#         # self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)
#
#         self.Up4 = up_conv(ch_in=512, ch_out=256)
#         self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
#         self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
#
#         self.Up3 = up_conv(ch_in=256, ch_out=256)
#         self.Att3 = Attention_block(F_g=256, F_l=128, F_int=64)
#         self.Up_conv3 = conv_block(ch_in=384, ch_out=256)
#
#         self.Up2 = up_conv(ch_in=256, ch_out=128)
#         self.Att2 = Attention_block(F_g=128, F_l=64, F_int=32)
#         self.Up_conv2 = conv_block(ch_in=192, ch_out=128)
#
#         # previous settings.
#         self.Up3 = up_conv(ch_in=256, ch_out=128)
#         self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
#         self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
#
#         self.Up2 = up_conv(ch_in=128, ch_out=64)
#         self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
#         self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
#         self.conv6 = conv_block(ch_in=64, ch_out=64)
#
#         self.mask_act = nn.ReLU(inplace=True)
#
#         if self.split_last:
#             self.Conv_1x1_f = nn.Conv2d(32, out_ch // 2, kernel_size=1, stride=1, padding=0)
#             self.Conv_1x1_b = nn.Conv2d(32, out_ch // 2, kernel_size=1, stride=1, padding=0)
#         else:
#             self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)
#
#     def forward(self, x):
#         # encoding path
#         x1 = self.Conv1(x)
#
#         x2 = self.Maxpool(x1)
#         x2 = self.Conv2(x2)
#
#         x3 = self.Maxpool(x2)
#         x3 = self.Conv3(x3)
#
#         x4 = self.Maxpool(x3)
#         x4 = self.Conv4(x4)
#
#         # x5 = self.Maxpool(x4)
#         # x5 = self.Conv5(x5)
#
#         # decoding + concat path
#         # d5 = self.Up5(x5)
#         # x4 = self.Att5(g=d5, x=x4)
#         # d5 = torch.cat((x4, d5), dim=1)
#         # d5 = self.Up_conv5(d5)
#
#         d4 = self.Up4(x4)
#         x3 = self.Att4(g=d4, x=x3)
#         d4 = torch.cat((x3, d4), dim=1)
#         d4 = self.Up_conv4(d4)
#
#         d3 = self.Up3(d4)
#         x2 = self.Att3(g=d3, x=x2)
#         d3 = torch.cat((x2, d3), dim=1)
#         d3 = self.Up_conv3(d3)
#
#         d2 = self.Up2(d3)
#         x1 = self.Att2(g=d2, x=x1)
#         d2 = torch.cat((x1, d2), dim=1)
#         d2 = self.Up_conv2(d2)
#         d2 = self.conv6(d2)
#
#         if self.split_last:
#             z1, z2 = torch.chunk(d2, chunks=2, dim=1)
#             z1_2 = self.Conv_1x1_f(z1)
#             z2_2 = self.Conv_1x1_b(z2)
#             d1 = torch.cat((z1_2, z2_2), dim=1)
#         else:
#             d1 = self.Conv_1x1(d2)
#         return d1, d2

class ATUNet_Color(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, loader_conf=None, split_last=True):
        super(ATUNet_Color, self).__init__()
        self.loader_conf = loader_conf
        self.split_last = split_last
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=256)
        self.Att3 = Attention_block(F_g=256, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=384, ch_out=256)

        self.Up2 = up_conv(ch_in=256, ch_out=128)
        self.Att2 = Attention_block(F_g=128, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=192, ch_out=128)

        # previous settings.
        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)
        self.conv6 = conv_block(ch_in=64, ch_out=64)

        self.mask_act = nn.ReLU(inplace=True)

        if self.split_last:
            self.Conv_1x1_f = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
            self.Conv_1x1_b = nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0)
            self.Conv_1x1_cb = nn.Conv2d(35, 3, kernel_size=1, stride=1, padding=0)
        else:
            self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        d2 = self.conv6(d2)

        if self.split_last:
            z1, z2 = torch.chunk(d2, chunks=2, dim=1)
            z1_2 = self.Conv_1x1_f(z1)
            z2_2 = self.Conv_1x1_b(z2)
            z3_2 = self.Conv_1x1_cb(torch.cat((x, z2), dim=1))
            d1 = torch.cat((z1_2, z2_2, z3_2), dim=1)
        else:
            d1 = self.Conv_1x1(d2)
        return d1

class ATUNetS(nn.Module):
    def __init__(self, in_ch=3, out_ch1=6, out_ch2=2):
        super(ATUNetS, self).__init__()

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(ch_in=in_ch, ch_out=64)
        self.Conv2 = conv_block(ch_in=64, ch_out=128)
        self.Conv3 = conv_block(ch_in=128, ch_out=256)
        self.Conv4 = conv_block(ch_in=256, ch_out=512)
        self.Conv5 = conv_block(ch_in=512, ch_out=1024)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1_1 = nn.Conv2d(32, out_ch1//2, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_2 = nn.Conv2d(32, out_ch1//2, kernel_size=1, stride=1, padding=0)

        self.Up5_2 = up_conv (ch_in=1024, ch_out=512)
        self.Att5_2 = Attention_block (F_g=512, F_l=512, F_int=256)
        self.Up_conv5_2 = conv_block (ch_in=1024, ch_out=512)

        self.Up4_2 = up_conv (ch_in=512, ch_out=256)
        self.Att4_2 = Attention_block (F_g=256, F_l=256, F_int=128)
        self.Up_conv4_2 = conv_block (ch_in=512, ch_out=256)

        self.Up3_2 = up_conv (ch_in=256, ch_out=128)
        self.Att3_2 = Attention_block (F_g=128, F_l=128, F_int=64)
        self.Up_conv3_2 = conv_block (ch_in=256, ch_out=128)

        self.Up2_2 = up_conv (ch_in=128, ch_out=64)
        self.Att2_2 = Attention_block (F_g=64, F_l=64, F_int=32)
        self.Up_conv2_2 = conv_block (ch_in=128, ch_out=64)

        self.Conv_1x1_3 = nn.Conv2d (32, out_ch2//2, kernel_size=1, stride=1, padding=0)
        self.Conv_1x1_4 = nn.Conv2d (32, out_ch2//2, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4_1 = self.Att5(g=d5, x=x4)
        d5 = torch.cat((x4_1, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        x3_1 = self.Att4(g=d4, x=x3)
        d4 = torch.cat((x3_1, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2_1 = self.Att3(g=d3, x=x2)
        d3 = torch.cat((x2_1, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1_1 = self.Att2(g=d2, x=x1)
        d2 = torch.cat((x1_1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        z1, z2 = torch.chunk(d2, chunks=2, dim=1)
        d1_z1 = self.Conv_1x1_1(z1)
        d1_z2 = self.Conv_1x1_2(z2)

        d5_2 = self.Up5_2 (x5)
        x4_2 = self.Att5_2 (g=d5, x=x4)
        d5_2 = torch.cat ((x4_2, d5_2), dim=1)
        d5_2 = self.Up_conv5_2 (d5_2)

        d4_2 = self.Up4_2 (d5_2)
        x3_2 = self.Att4_2 (g=d4_2, x=x3)
        d4_2 = torch.cat ((x3_2, d4_2), dim=1)
        d4_2 = self.Up_conv4 (d4_2)

        d3_2 = self.Up3_2 (d4_2)
        x2_2 = self.Att3 (g=d3_2, x=x2)
        d3_2 = torch.cat ((x2_2, d3_2), dim=1)
        d3_2 = self.Up_conv3 (d3_2)

        d2_2 = self.Up2 (d3_2)
        x1_2 = self.Att2 (g=d2_2, x=x1)
        d2_2 = torch.cat ((x1_2, d2_2), dim=1)
        d2_2 = self.Up_conv2 (d2_2)

        z3, z4 = torch.chunk(d2_2, chunks=2, dim=1)

        d2_z3 = self.Conv_1x1_3 (z3)
        d2_z4 = self.Conv_1x1_4 (z4)

        return torch.cat([d1_z1, d1_z2, d2_z3, d2_z4], dim=1)


class ATUNetM(nn.Module):
    def __init__(self, in_ch1=6, in_ch2=6, out_ch=2,
                 split_last=True):
        super(ATUNetM, self).__init__()
        self.split_last = split_last
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1_1 = conv_block(ch_in=in_ch1, ch_out=32)
        self.Conv2_1 = conv_block(ch_in=32, ch_out=64)
        self.Conv3_1 = conv_block(ch_in=64, ch_out=128)
        self.Conv4_1 = conv_block(ch_in=128, ch_out=256)
        self.Conv5_1 = conv_block(ch_in=256, ch_out=512)

        self.Conv1_2 = conv_block (ch_in=in_ch2, ch_out=32)
        self.Conv2_2 = conv_block (ch_in=32, ch_out=64)
        self.Conv3_2 = conv_block (ch_in=64, ch_out=128)
        self.Conv4_2 = conv_block (ch_in=128, ch_out=256)
        self.Conv5_2 = conv_block (ch_in=256, ch_out=512)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Att5 = Attention_block(F_g=512, F_l=512, F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Att4 = Attention_block(F_g=256, F_l=256, F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Att3 = Attention_block(F_g=128, F_l=128, F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Att2 = Attention_block(F_g=64, F_l=64, F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        if self.split_last:
            self.Conv_1x1_f = nn.Conv2d(32, out_ch // 2, kernel_size=1, stride=1, padding=0)
            self.Conv_1x1_b = nn.Conv2d(32, out_ch // 2, kernel_size=1, stride=1, padding=0)
        else:
            self.Conv_1x1 = nn.Conv2d(64, out_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x, y):
        # encoding path
        x1 = self.Conv1_1(x)
        x2 = self.Maxpool(x1)
        x2 = self.Conv2_1(x2)
        x3 = self.Maxpool(x2)
        x3 = self.Conv3_1(x3)
        x4 = self.Maxpool(x3)
        x4 = self.Conv4_1(x4)
        x5 = self.Maxpool(x4)
        x5 = self.Conv5_1(x5)

        y1 = self.Conv1_2(y)
        y2 = self.Maxpool(y1)
        y2 = self.Conv2_2(y2)
        y3 = self.Maxpool(y2)
        y3 = self.Conv3_2(y3)
        y4 = self.Maxpool(y3)
        y4 = self.Conv4_2(y4)
        y5 = self.Maxpool(y4)
        y5 = self.Conv5_2(y5)

        d_shared = torch.cat([x5, y5], dim=1)

        d5_1 = self.Up5(d_shared)
        x4_1 = self.Att5(g=d5_1, x=torch.cat([x4, y4], dim=1))
        d5_1 = torch.cat([x4_1, d5_1], dim=1)
        d5_1 = self.Up_conv5(d5_1)

        d4_1 = self.Up4(d5_1)
        x3_1 = self.Att4(g=d4_1, x=torch.cat([x3, y3], dim=1))
        d4_1 = torch.cat([x3_1, d4_1], dim=1)
        d4_1 = self.Up_conv4(d4_1)

        d3_1 = self.Up3(d4_1)
        x2_1 = self.Att3(g=d3_1, x=torch.cat([x2, y2], dim=1))
        d3_1 = torch.cat([x2_1, d3_1], dim=1)
        d3_1 = self.Up_conv3(d3_1)

        d2_1 = self.Up2(d3_1)
        x1_1 = self.Att2(g=d2_1, x=torch.cat([x1, y1], dim=1))
        d2_1 = torch.cat([x1_1, d2_1], dim=1)
        d2_1 = self.Up_conv2(d2_1)

        if self.split_last:
            z1, z2 = torch.chunk(d2_1, chunks=2, dim=1)
            z1_2 = self.Conv_1x1_f(z1)
            z2_2 = self.Conv_1x1_b(z2)
            d1_1 = torch.cat((z1_2, z2_2), dim=1)
        else:
            d1_1 = self.Conv_1x1(d2_1)

        return d1_1


def weight_init_basic(m):
    if isinstance (m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_ (0, math.sqrt (2. / n))
        # m.weight.data.uniform_ (-math.sqrt (6. / n), math.sqrt (6. / n))
    elif isinstance (m, nn.BatchNorm2d):
        m.weight.data.fill_ (1)
        m.bias.data.zero_ ()
    elif isinstance (m, nn.Linear):
        m.bias.data.zero_ ()
    return m


if __name__ == '__main__':
    input1 = Variable (torch.randn (4, 6, 256, 128)).float()
    input2 = Variable(torch.randn(4, 6, 256, 128)).float()
    latent = Variable (torch.randn (4, 1024, 16, 8)).float()
    target = Variable (torch.randn (4, 128, 256, 256)).float()

    model = ATUNetM(6, 6, 2)
    output = model(input1, input2)
    print(output.size())
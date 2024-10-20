import torch.nn as nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import sys
class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None):
        super(BasicConvBlock, self).__init__()
        
        if out_channels is None:
                out_channels = in_channels
        
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, padding=1),
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d( in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d( out_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )        
    
    def forward(self,x):
        x=self.conv(x)
        return x

class Conv2dStaticSamePadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, groups=1, dilation=1, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=bias, groups=groups)
        self.stride = self.conv.stride
        self.kernel_size = self.conv.kernel_size
        self.dilation = self.conv.dilation

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]
        
        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.conv(x)
        return x

class MaxPool2dStaticSamePadding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pool = nn.MaxPool2d(*args, **kwargs)
        self.stride = self.pool.stride
        self.kernel_size = self.pool.kernel_size

        if isinstance(self.stride, int):
            self.stride = [self.stride] * 2
        elif len(self.stride) == 1:
            self.stride = [self.stride[0]] * 2

        if isinstance(self.kernel_size, int):
            self.kernel_size = [self.kernel_size] * 2
        elif len(self.kernel_size) == 1:
            self.kernel_size = [self.kernel_size[0]] * 2

    def forward(self, x):
        h, w = x.shape[-2:]
        
        extra_h = (math.ceil(w / self.stride[1]) - 1) * self.stride[1] - w + self.kernel_size[1]
        extra_v = (math.ceil(h / self.stride[0]) - 1) * self.stride[0] - h + self.kernel_size[0]

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        x = self.pool(x)
        return x

class SSFC(torch.nn.Module):
    def __init__(self, in_ch):
        super(SSFC, self).__init__()
        # self.proj = nn.Conv2d(in_ch, in_ch, kernel_size=1)  # generate k by conv

    def forward(self, x, y):
        _, _, h, w = x.size()

        q = y.mean(dim=[2, 3], keepdim=True)
        # k = self.proj(x)
        k = x
        square = (k - q).pow(2)
        sigma = square.sum(dim=[2, 3], keepdim=True) / (h * w)
        att_score = square / (2 * sigma + np.finfo(np.float32).eps) + 0.5
        att_weight = nn.Sigmoid()(att_score)
        # print(sigma)

        return x * att_weight

class ConvBlock(nn.Module):
    def __init__(self, num_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1, groups=num_channels),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(num_features=num_channels, momentum=0.9997, eps=4e-5), nn.ReLU())

    def forward(self, input):
        return self.conv(input)
    
class BiFPN(nn.Module):
    def __init__(self, num_channels, conv_channels, epsilon=1e-4):
        super(BiFPN, self).__init__()
        self.epsilon = epsilon

        self.conv5 = BasicConvBlock(num_channels)
        self.conv4 = BasicConvBlock(num_channels)
        self.conv3 = BasicConvBlock(num_channels)
        self.conv2 = BasicConvBlock(num_channels)
        self.conv1 = BasicConvBlock(num_channels)

        self.conv5_1 = BasicConvBlock(num_channels)
        self.conv4_1 = BasicConvBlock(num_channels)
        self.conv3_1 = BasicConvBlock(num_channels)
        self.conv2_1 = BasicConvBlock(num_channels)
        self.conv1_1 = BasicConvBlock(num_channels)

        self.conv1_down = BasicConvBlock(num_channels)
        self.conv2_down = BasicConvBlock(num_channels)
        self.conv3_down = BasicConvBlock(num_channels)
        self.conv4_down = BasicConvBlock(num_channels)
        self.conv5_down = BasicConvBlock(num_channels)

        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p1_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p2_upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.p1_upsample_1 = nn.Upsample(scale_factor=2, mode='nearest')

        self.p2_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p3_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p4_downsample = MaxPool2dStaticSamePadding(3, 2)
        self.p5_downsample = MaxPool2dStaticSamePadding(3, 2)

        # Channel compression layers
        self.p5_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[4], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.p4_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[3], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.p3_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[2], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.p2_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[1], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )
        self.p1_down_channel = nn.Sequential(
            Conv2dStaticSamePadding(conv_channels[0], num_channels, 1),
            nn.BatchNorm2d(num_channels, momentum=0.01, eps=1e-3),
            nn.ReLU(inplace=True),
        )

        self.csac_p1_0=SSFC(num_channels)
        self.csac_p2_0=SSFC(num_channels)
        self.csac_p3_0=SSFC(num_channels)
        self.csac_p4_0=SSFC(num_channels)
        self.csac_p5_0=SSFC(num_channels)

        self.csac_p1_1=SSFC(num_channels)
        self.csac_p2_1=SSFC(num_channels)
        self.csac_p3_1=SSFC(num_channels)
        self.csac_p4_1=SSFC(num_channels)
        self.csac_p5_1=SSFC(num_channels)

        self.csac_p1_2=SSFC(num_channels)
        self.csac_p2_2=SSFC(num_channels)
        self.csac_p3_2=SSFC(num_channels)
        self.csac_p4_2=SSFC(num_channels)

        self.csac_p51_0=SSFC(num_channels)
        self.csac_p41_0=SSFC(num_channels)
        self.csac_p31_0=SSFC(num_channels)
        self.csac_p21_0=SSFC(num_channels)
        
        self.csac_p51_1=SSFC(num_channels)
        self.csac_p41_1=SSFC(num_channels)
        self.csac_p31_1=SSFC(num_channels)
        self.csac_p21_1=SSFC(num_channels)

        self.csac_p52_0=SSFC(num_channels)
        self.csac_p42_0=SSFC(num_channels)
        self.csac_p32_0=SSFC(num_channels)
        self.csac_p22_0=SSFC(num_channels)
        self.csac_p12_0=SSFC(num_channels)

        self.csac_p52_1=SSFC(num_channels)
        self.csac_p42_1=SSFC(num_channels)
        self.csac_p32_1=SSFC(num_channels)
        self.csac_p22_1=SSFC(num_channels)
        self.csac_p12_1=SSFC(num_channels)

        self.csac_p42_2=SSFC(num_channels)
        self.csac_p32_2=SSFC(num_channels)
        self.csac_p22_2=SSFC(num_channels)
        self.csac_p12_2=SSFC(num_channels)

        self.Classifier = Classifier(num_channels, 2)

    def forward(self, inputs):
        p1_in, p2_in, p3_in, p4_in, p5_in = inputs

        p1_in = self.p1_down_channel(p1_in)
        p2_in = self.p2_down_channel(p2_in)
        p3_in = self.p3_down_channel(p3_in)
        p4_in = self.p4_down_channel(p4_in)
        p5_in = self.p5_down_channel(p5_in)

        # up
        p5_in=self.conv5(p5_in)
        p4_in=self.conv4(p4_in + self.p4_upsample(p5_in))
        p3_in=self.conv3(p3_in + self.p3_upsample(p4_in))
        p2_in=self.conv2(p2_in + self.p2_upsample(p3_in))
        p1_in=self.conv1(p1_in + self.p1_upsample(p2_in))

        out_1 = self.Classifier(p1_in)

        # down
        p2_1 = self.conv2_down(self.csac_p21_0(p2_in, p1_in) + self.csac_p21_1(self.p2_downsample(p1_in), p1_in))
        p3_1 = self.conv3_down(self.csac_p31_0(p3_in, p1_in) + self.csac_p31_1(self.p3_downsample(p2_1), p1_in))
        p4_1 = self.conv4_down(self.csac_p41_0(p4_in, p1_in) + self.csac_p41_1(self.p4_downsample(p3_1), p1_in))
        p5_1 = self.conv5_down(self.csac_p51_0(p5_in, p1_in) + self.csac_p51_1(self.p5_downsample(p4_1), p1_in))

        # up
        p4_2 = self.conv4_1(self.csac_p42_0(p4_in, p1_in) + self.csac_p42_1(p4_1, p1_in) + self.csac_p42_2(self.p4_upsample_1(p5_1), p1_in))
        p3_2 = self.conv3_1(self.csac_p32_0(p3_in, p1_in) + self.csac_p32_1(p3_1, p1_in) + self.csac_p32_2(self.p3_upsample_1(p4_2), p1_in))
        p2_2 = self.conv2_1(self.csac_p22_0(p2_in, p1_in) + self.csac_p22_1(p2_1, p1_in) + self.csac_p22_2(self.p2_upsample_1(p3_2), p1_in))
        p1_2 = self.conv1_1(self.csac_p12_0(p1_in, p1_in) + self.csac_p12_1(p1_in, p1_in) + self.csac_p12_2(self.p1_upsample_1(p2_2), p1_in))

        out_2 = self.Classifier(p1_2)
        return out_1, out_2

class Classifier(nn.Module):
    def __init__(self, in_ch, classes):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(
            in_ch, in_ch//4, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(
            in_ch//4, in_ch//8, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(
            in_ch//8, classes*4, kernel_size=1)

        self.ps3 = nn.PixelShuffle(2)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.ps3(x)
        
        return x
    
if __name__=='__main__':
    model = BiFPN(128,[64,64,128,256,512])
    x_1 = torch.randn(8, 64, 128, 128)
    x_2 = torch.randn(8, 64, 64, 64)
    x_3 = torch.randn(8, 128, 32, 32)
    x_4 = torch.randn(8, 256, 16, 16)
    x_5 = torch.randn(8, 512, 8, 8)
    x = [x_1, x_2, x_3, x_4, x_5]
    x = model(x)
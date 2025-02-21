import torch
import torch.nn as nn
from torch import Tensor, reshape, stack
from torch.nn import (
    Conv2d,
    InstanceNorm2d,
    Module,
    PReLU,
    Sequential,
    Upsample,
)
# from .hfam import HFAM
from .sge import SpatialGroupEnhance
from .bam import BAM
# from .vertical import VerticalFusion

class MSFF(nn.Module):
    def __init__(self, inchannel, mid_channel):
        super(MSFF, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 5, stride=1, padding=2, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(inchannel, mid_channel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, mid_channel, 7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(mid_channel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(mid_channel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))
        self.convmix = nn.Sequential(nn.Conv2d(4 * inchannel, inchannel, 1, stride=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inchannel, inchannel, 3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(inchannel),
                                   nn.ReLU(inplace=True))

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x_f = torch.cat([x1, x2, x3, x4], dim=1)
        out = self.convmix(x_f)
        return out

class MDFM(nn.Module):
    def __init__(self, in_d, out_d):
        super(MDFM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d
        self.MPFL = MSFF(inchannel=in_d, mid_channel=64)   ##64
        # self.sge = SpatialGroupEnhance(2)
        # self.diff = DiffEnhanceBlock(in_d)

        self.conv_diff_enh = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

        self.conv_sub = nn.Sequential(
            nn.Conv2d(self.in_d, self.in_d, 3,  padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

        self.convmix = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 3, groups=self.in_d, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

        '''
        self.concat = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_d, self.in_d, 1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
            nn.Sigmoid()
        )
        '''

        self.x_concat = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.in_d, self.in_d, 1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

        self.bam = BAM(self.in_d)
        # self.hfam = HFAM()
        # self.deform = VerticalFusion(out_d, num_heads=num_heads, num_points=num_points, kernel_layers=1)

    def forward(self, x1, x2):
        # difference enhance
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x_sub = torch.abs(x1 - x2)
        x_cat = self.x_concat(torch.cat([x1, x2], dim = 1))
        x_att = self.bam(x_sub, x_cat)
        x1 = (x1 * x_att) # + self.conv_diff_enh(x1)
        x2 = (x2 * x_att) # + self.conv_diff_enh(x2)
        # fusion
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)
        # after ca
        x_f = x_f * x_att
        out = self.conv_dr(x_f)

        '''
        # difference enhance
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x_sub = torch.abs(x1 - x2)
        # x_att = torch.sigmoid(x_sub)
        x_cat = self.x_concat(torch.cat([x1, x2], dim = 1))
        x_att = self.bam(x_cat, x_sub)
        # print(x_cat.shape)
        # x_cat_att = self.concat(x_cat)
        # print(x_cat_att.shape)
        # x1 = (x1 * x_att) + x_cat_att * self.conv_diff_enh(x1)
        # x2 = (x2 * x_att) + x_cat_att * self.conv_diff_enh(x2)
        x1 = (x1 * x_att) + x1
        x2 = (x2 * x_att) + x2

        # x1 = (x1 * x_att) + self.MPFL(self.conv_diff_enh(x1))
        # x2 = (x2 * x_att) + self.MPFL(self.conv_diff_enh(x2))
        # x1_ = x1.clone()
        # x2_ = x2.clone()
        # x1 = (x1 * x_att) + self.deform(x1, x2_)
        # x2 = (x2 * x_att) + self.deform(x2, x1_)
        # x1 = (x1 * x_att) + self.conv_diff_enh(x1)
        # x2 = (x2 * x_att) + self.conv_diff_enh(x2)
        # fusion
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        # x_f = self.sge(x_f)
        x_f = self.convmix(x_f)
        # after ca
        x_f = x_f * x_att
        out = self.conv_dr(x_f)
        '''

        '''
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x1 = self.MPFL(self.conv_diff_enh(x1))
        x2 = self.MPFL(self.conv_diff_enh(x2))
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)
        out = self.conv_dr(x_f)
        # out = self.diff(x1, x2)
        
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        # x_f = self.sge(x_f)
        x_f = self.convmix(x_f)
        out = self.conv_dr(x_f)
        '''
        return out

if __name__ == '__main__':
    x1 = torch.randn((32, 128, 8, 8))
    x2 = torch.randn((32, 128, 8, 8))
    model = MDFM(128, 128)
    out = model(x1, x2)
    print(out.shape)
# (phase: test) acc: 99.186 miou: 92.122 mf1: 95.760 iou_0: 99.146 iou_1: 85.098 F1_0: 99.571 F1_1: 91.949 precision_0: 99.531 precision_1: 92.650 recall_0: 99.611 recall_1: 91.258 
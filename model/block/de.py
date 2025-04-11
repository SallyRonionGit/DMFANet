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
from .bam import BAM

class MDFM(nn.Module):
    def __init__(self, in_d, out_d):
        super(MDFM, self).__init__()
        self.in_d = in_d
        self.out_d = out_d

        self.conv_dr = nn.Sequential(
            nn.Conv2d(self.in_d, self.out_d, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.out_d),
            nn.ReLU(inplace=True)
        )

        self.convmix = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 3, groups=self.in_d, padding=1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True),
        )

        self.x_concat = nn.Sequential(
            nn.Conv2d(2 * self.in_d, self.in_d, 1, bias=False),
            nn.BatchNorm2d(self.in_d),
            nn.ReLU(inplace=True)
        )

        self.bam = BAM(self.in_d)

    def forward(self, x1, x2):
        b, c, h, w = x1.shape[0], x1.shape[1], x1.shape[2], x1.shape[3]
        x_sub = torch.abs(x1 - x2)
        x_cat = self.x_concat(torch.cat([x1, x2], dim = 1))
        x_att = self.bam(x_sub, x_cat)
        x1 = (x1 * x_att) 
        x2 = (x2 * x_att) 
        x_f = torch.stack((x1, x2), dim=2)
        x_f = torch.reshape(x_f, (b, -1, h, w))
        x_f = self.convmix(x_f)
        x_f = x_f * x_att
        out = self.conv_dr(x_f)
        return out

if __name__ == '__main__':
    x1 = torch.randn((32, 128, 8, 8))
    x2 = torch.randn((32, 128, 8, 8))
    model = MDFM(128, 128)
    out = model(x1, x2)
    print(out.shape)

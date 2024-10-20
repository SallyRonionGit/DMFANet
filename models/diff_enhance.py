import torch
from torch import nn
import math
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
import torch.nn.functional as F  
import sys

class DiffEnhanceBlock(nn.Module):  
    def __init__(self, in_channels, groups=8):  
        super(DiffEnhanceBlock, self).__init__()  
        self.concat_branch = ConcatBlock(in_channels)
        self.mixed_branch = MixingBlock(2*in_channels, in_channels)
        self.diff_fusion = GFF(in_channels, in_channels)

    def forward(self, x1, x2):  
        sub_branch = torch.abs(x1 - x2)
        channel_branch = self.concat_branch(x1, x2) 
        spatial_branch = self.mixed_branch(x1, x2) 
        out = self.diff_fusion(sub_branch, channel_branch, spatial_branch)
        return out

# Channel-domain
class ConcatBlock(nn.Module):
    def __init__(self,in_channels):
        super(ConcatBlock, self).__init__()  
        self.concat_block = nn.Sequential(  
            nn.Conv2d(2*in_channels, 2*in_channels, kernel_size=3, padding=1),  
            nn.BatchNorm2d(2*in_channels),
            nn.ReLU(inplace=True),  

            nn.Conv2d(2*in_channels, in_channels, kernel_size=1),  
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True) 
        )

    def forward(self, x1, x2):  
        x = torch.cat((x1, x2), dim=1)  
        x = self.concat_block(x)  
        return x

# Temporal-domain
class MixingBlock(Module):
    def __init__(
        self,
        ch_in: int,
        ch_out: int,
    ):
        super().__init__()
        self._convmix = Sequential(
            Conv2d(ch_in, ch_out, 3, groups=ch_out, padding=1),
            PReLU(),
            InstanceNorm2d(ch_out),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        mixed = stack((x, y), dim=2)
        mixed = reshape(mixed, (x.shape[0], -1, x.shape[2], x.shape[3]))
        return self._convmix(mixed)

class GatedFullyFusion(nn.Module):  
    def __init__(self, in_channels, out_channels):  
        super(GatedFullyFusion, self).__init__()  
        self.gated_fully_fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ) 

    def forward(self, x):  
        x = self.gated_fully_fusion(x) 
        return x

class GFF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GFF, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.gate_fusion = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ) 

    def forward(self, x_1, x_2, x_3):
        gate_1 = self.gate(x_1)
        gate_2 = self.gate(x_2)
        gate_3 = self.gate(x_3) 
        x_gff = (1+gate_1)*x_1 + (1-gate_1)*(gate_2*x_2 + gate_3*x_3)
        x_gff =  self.gate_fusion(x_gff)
        return x_gff

if __name__=='__main__':
    model = DiffEnhanceBlock(128)
    x1 = torch.randn(8, 128, 64, 64)
    x2 = torch.randn(8, 128, 64, 64)
    out = model(x1, x2)
    print(out.shape)

 













    






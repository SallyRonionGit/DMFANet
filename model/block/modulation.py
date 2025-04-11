import torch
import torch.nn as nn
import torch.nn.functional as F
from .convs import ConvBnRelu
from .focal import FocalModulation

class VerticalFusion(nn.Module):
    def __init__(self, channels, focal_window=3, focal_level=4, kernel_layers=1, up_kernel_size=5, enc_kernel_size=3):
        super(VerticalFusion, self).__init__()
        self.selfattn = FocalModulation(dim=channels, focal_window=3, focal_level=4, focal_factor=2, bias=True, 
                                        proj_drop=0., use_postln_in_modulation=False, normalize_modulator=False)
        convs = []
        convs.append(ConvBnRelu(in_channels=channels, out_channels=channels))
        for _ in range(kernel_layers - 1):
            convs.append(ConvBnRelu(in_channels=channels, out_channels=channels))
        self.convs = nn.Sequential(*convs)
        self.enc = ConvBnRelu(channels, up_kernel_size ** 2, kernel_size=enc_kernel_size,
                              stride=1, padding=enc_kernel_size // 2, dilation=1)

        self.upsmp = nn.Upsample(scale_factor=2, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=up_kernel_size, dilation=2,
                                padding=up_kernel_size // 2 * 2)

    def forward(self, x1, x2):
        B, C, H, W = x1.size()
        _, _, H2, W2 = x2.size()
        x1_, x2_ = x1.clone(), x2.clone()
        x = self.selfattn(x1_)
        kernel = self.convs(x2_)
        kernel = self.enc(kernel)
        kernel = F.softmax(kernel, dim=1)
        x = F.interpolate(x, size=(H2, W2), mode='nearest')
        x = self.unfold(x)
        x = x.view(B, C, -1, H2, W2)
        fuse = torch.einsum('bkhw,bckhw->bchw', [kernel, x])
        fuse += x2_
        return fuse

if __name__ == '__main__':
    x1 = torch.randn(8, 128, 8, 8)
    x2 = torch.randn(8, 128, 8, 8)
    model = VerticalFusion(128)
    out = model(x1, x2)
    print(out.shape)
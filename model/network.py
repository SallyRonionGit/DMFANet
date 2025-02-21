import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from .backbone.mobilenetv2 import mobilenet_v2
from .block.fpn import FPN
from .block.modulation import VerticalFusion
from .block.convs import ConvBnRelu, DsBnRelu
from .util import init_method
from .block.heads import FCNHead, GatedResidualUpHead

from .block.de import MDFM

# from .block.rfe import C3RFEMDE
# from .block.arm import AmbiguityRefinementModule
# from .block.lae import LAEDE
# from .block.SimCSPSPPF import SimCSPSPPFDE
# from .block.msf import MSFDE
# from .block.gdm import MDFM

def get_backbone(backbone_name):
    if backbone_name == 'mobilenetv2':
        backbone = mobilenet_v2(pretrained=True, progress=True)
        backbone.channels = [16, 24, 32, 96, 320]
    elif backbone_name == 'resnet18d':
        backbone = timm.create_model('resnet18d', pretrained=True, features_only=True)
        backbone.channels = [64, 64, 128, 256, 512]
    else:
        raise NotImplementedError("BACKBONE [%s] is not implemented!\n" % backbone_name)
    return backbone


def get_fpn(fpn_name, in_channels, out_channels, deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv'):
    if fpn_name == 'fpn':
        fpn = FPN(in_channels, out_channels, deform_groups, gamma_mode, beta_mode)
    else:
        raise NotImplementedError("FPN [%s] is not implemented!\n" % fpn_name)
    return fpn


class Detector(nn.Module):
    def __init__(self, backbone_name='mobilenetv2', fpn_name='fpn', fpn_channels=128,
                 deform_groups=4, gamma_mode='SE', beta_mode='contextgatedconv',
                 num_heads=1, num_points=8, kernel_layers=1, dropout_rate=0.1, init_type='kaiming_normal'):
        super().__init__()
        self.backbone = get_backbone(backbone_name)
        self.fpn = get_fpn(fpn_name, in_channels=self.backbone.channels[-4:], out_channels=fpn_channels,
                           deform_groups=deform_groups, gamma_mode=gamma_mode, beta_mode=beta_mode)
        
        '''
        self.de_p5_to_p4 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=4,
                                                    kernel_layers=kernel_layers)
        self.de_p4_to_p3 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=8,
                                                    kernel_layers=kernel_layers)
        self.de_p3_to_p2 = VerticalFusion(fpn_channels, num_heads=num_heads, num_points=16,
                                                    kernel_layers=kernel_layers)
        '''

        self.p5_to_p4 = VerticalFusion(fpn_channels, focal_window=5, focal_level=3)
        self.p4_to_p3 = VerticalFusion(fpn_channels, focal_window=3, focal_level=3)
        self.p3_to_p2 = VerticalFusion(fpn_channels, focal_window=1, focal_level=3)
        
        '''
        self.diff_rfe_p2 = C3RFEMDE(128, 128)
        self.diff_rfe_p3 = C3RFEMDE(128, 128)
        self.diff_rfe_p4 = C3RFEMDE(128, 128)
        self.diff_rfe_p5 = C3RFEMDE(128, 128)
        '''

        '''
        self.diff_lae_p2 = LAEDE(128, 128)
        self.diff_lae_p3 = LAEDE(128, 128)
        self.diff_lae_p4 = LAEDE(128, 128)
        self.diff_lae_p5 = LAEDE(128, 128)
        '''

        '''
        self.diff_simcspsppf_p2 = SimCSPSPPFDE(128, 128)
        self.diff_simcspsppf_p3 = SimCSPSPPFDE(128, 128)
        self.diff_simcspsppf_p4 = SimCSPSPPFDE(128, 128)
        self.diff_simcspsppf_p5 = SimCSPSPPFDE(128, 128)
        '''

        '''
        self.diff_msf_p2 = MSFDE(128, 128)
        self.diff_msf_p3 = MSFDE(128, 128)
        self.diff_msf_p4 = MSFDE(128, 128)
        self.diff_msf_p5 = MSFDE(128, 128)
        '''

        self.diff_mdfm_p2 = MDFM(128, 128)
        self.diff_mdfm_p3 = MDFM(128, 128)
        self.diff_mdfm_p4 = MDFM(128, 128)
        self.diff_mdfm_p5 = MDFM(128, 128)
        
        '''
        self.diff_arm_p2 = AmbiguityRefinementModule(128, 128, 64)
        self.diff_arm_p3 = AmbiguityRefinementModule(128, 128, 32)
        self.diff_arm_p4 = AmbiguityRefinementModule(128, 128, 16)
        self.diff_arm_p5 = AmbiguityRefinementModule(128, 128, 8)
        '''

        self.p5_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p4_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p3_head = nn.Conv2d(fpn_channels, 2, 1)
        self.p2_head = nn.Conv2d(fpn_channels, 2, 1)
        self.project = nn.Sequential(nn.Conv2d(fpn_channels*4, fpn_channels, 1, bias=False),
                                     nn.BatchNorm2d(fpn_channels),
                                     nn.ReLU(True)
                                     )
        self.head = GatedResidualUpHead(fpn_channels, 2, dropout_rate=dropout_rate)
        # init_method(self.fpn, self.p5_to_p4, self.p4_to_p3, self.p3_to_p2, self.p5_head, self.p4_head,
        #             self.p3_head, self.p2_head, init_type=init_type)

    def forward(self, x1, x2):
        ### Extract backbone features
        t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.backbone.forward(x1)
        t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.backbone.forward(x2)
        t1_p2, t1_p3, t1_p4, t1_p5 = self.fpn([t1_c2, t1_c3, t1_c4, t1_c5])
        t2_p2, t2_p3, t2_p4, t2_p5 = self.fpn([t2_c2, t2_c3, t2_c4, t2_c5])
        
        '''
        diff_p2 = torch.abs(t1_p2 - t2_p2)
        diff_p3 = torch.abs(t1_p3 - t2_p3)
        diff_p4 = torch.abs(t1_p4 - t2_p4)
        diff_p5 = torch.abs(t1_p5 - t2_p5)
        '''

        '''
        diff_p2 = self.diff_rfe_p2(t1_p2, t2_p2)
        diff_p3 = self.diff_rfe_p3(t1_p3, t2_p3)
        diff_p4 = self.diff_rfe_p4(t1_p4, t2_p4)
        diff_p5 = self.diff_rfe_p5(t1_p5, t2_p5)
        '''

        '''
        diff_p2 = self.diff_lae_p2(t1_p2, t2_p2)
        diff_p3 = self.diff_lae_p3(t1_p3, t2_p3)
        diff_p4 = self.diff_lae_p4(t1_p4, t2_p4)
        diff_p5 = self.diff_lae_p5(t1_p5, t2_p5)
        '''

        '''
        diff_p2 = self.diff_simcspsppf_p2(t1_p2, t2_p2)
        diff_p3 = self.diff_simcspsppf_p3(t1_p3, t2_p3)
        diff_p4 = self.diff_simcspsppf_p4(t1_p4, t2_p4)
        diff_p5 = self.diff_simcspsppf_p5(t1_p5, t2_p5)
        '''

        '''
        diff_p2 = self.diff_msf_p2(t1_p2, t2_p2)
        diff_p3 = self.diff_msf_p3(t1_p3, t2_p3)
        diff_p4 = self.diff_msf_p4(t1_p4, t2_p4)
        diff_p5 = self.diff_msf_p5(t1_p5, t2_p5)
        '''

        diff_p2 = self.diff_mdfm_p2(t1_p2, t2_p2)
        diff_p3 = self.diff_mdfm_p3(t1_p3, t2_p3)
        diff_p4 = self.diff_mdfm_p4(t1_p4, t2_p4)
        diff_p5 = self.diff_mdfm_p5(t1_p5, t2_p5)

        '''
        diff_p2 = self.diff_arm_p2(t1_p2, t2_p2)
        diff_p3 = self.diff_arm_p3(t1_p3, t2_p3)
        diff_p4 = self.diff_arm_p4(t1_p4, t2_p4)
        diff_p5 = self.diff_arm_p5(t1_p5, t2_p5)
        '''
        
        fea_p5 = diff_p5
        pred_p5 = self.p5_head(fea_p5)
        fea_p4 = self.p5_to_p4(fea_p5, diff_p4)
        pred_p4 = self.p4_head(fea_p4)
        fea_p3 = self.p4_to_p3(fea_p4, diff_p3)
        pred_p3 = self.p3_head(fea_p3)
        fea_p2 = self.p3_to_p2(fea_p3, diff_p2)
        pred_p2 = self.p2_head(fea_p2)
        pred = self.head(fea_p2)

        pred_p2 = F.interpolate(pred_p2, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p3 = F.interpolate(pred_p3, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p4 = F.interpolate(pred_p4, size=(256, 256), mode='bilinear', align_corners=False)
        pred_p5 = F.interpolate(pred_p5, size=(256, 256), mode='bilinear', align_corners=False)

        return pred, pred_p2, pred_p3, pred_p4, pred_p5


if __name__ == '__main__':
    x1 = torch.randn((32, 512, 8, 8))
    x2 = torch.randn((32, 512, 8, 8))
    model = Detector(512, 512)
    out = model(x1, x2)
    print(out.shape)

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
from torch.optim import lr_scheduler
import numpy as np
import os
import sys
import cv2  
import functools
import models
from models.diff_enhance import DiffEnhanceBlock
from models.DSDE import BiFPN
from compare.FC_EF import Unet
from compare.FC_Siam_conc import SiamUnet_conc
from compare.FC_Siam_diff import SiamUnet_diff
from compare.NestedUNet import NestedUNet
from compare.SNUNet import SNUNet_ECAM
from compare.DTCDSCN import CDNet_model
from compare.ChangeFormer import ChangeFormerV6
from compare.A2Net import A2Net
from compare.DMINet import DMINet
from compare.IFNet import DSIFN
from compare.TFI_GR import TFI_GR
from compare.BIT import BASE_Transformer
from compare.SEIFNet import SEIFNet
from compare.USSFC import USSFCNet
from compare.HFANet import HFANet

def get_scheduler(optimizer, args, total_steps=None, batch_id=None):
    if args.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - epoch / float(args.max_epochs + 1)
            # epoch lr_poly
            # lr_l = args.lr * (1.0 - epoch / float(args.max_epochs + 1)) ** 0.9
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif args.lr_policy == 'step':
        step_size = args.max_epochs//3
        # args.lr_decay_iters
        # lr*gamma after step_size (1/3)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.1)
        # HFANet
        # milestone = [10, 15, 30, 40]
        # scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestone)
        # scheduler = lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', args.lr_policy)   
    return scheduler

# residual identity
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """
    Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.

    BatchNorm : affine parameters(gamma,beta) track_running_stats=True(track when inference)
    InstanceNorm : calculate each channel of the image to get std and var and normalization
                    single image rather than batch normalization
    class Identity is used to generate Identity Mapping which means no need to normalization
    """
    # functools.partial to fix parameter
    # affine : gamma beta
    # track_running_stats : track mean and std
    # return function object
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights include BatchNorm Convolution and Linear.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  

def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """
    Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        if len(gpu_ids) > 1:
            net = torch.nn.DataParallel(net, gpu_ids)  
    init_weights(net, init_type, init_gain=init_gain)
    return net

def define_G(args, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if args.net_G == 'FC_EF':
        net = Unet(input_nbr=3, label_nbr=2)
    elif args.net_G == 'FC_Siam_conc':
        net = SiamUnet_conc(input_nbr=3, label_nbr=2)
    elif args.net_G == 'FC_Siam_diff':
        net = SiamUnet_diff(input_nbr=3, label_nbr=2)
    elif args.net_G == 'UNet++':
        net = NestedUNet(num_classes=2, input_channels=6, deep_supervision=True)
    elif args.net_G == 'SNUNet':
        net = SNUNet_ECAM(in_ch=3, out_ch=2)
    elif args.net_G == 'DTCDSCN':
        net = CDNet_model(in_channels=3)
    elif args.net_G == 'ChangeFormer':
        net = ChangeFormerV6()
    elif args.net_G == 'A2Net':
        net = A2Net(input_nc=3, output_nc=2)
    elif args.net_G == 'DMINet':
        net = DMINet(pretrained=True)
    elif args.net_G == 'IFNet':
        net = DSIFN()
    elif args.net_G == 'TFI-GR':
        net = TFI_GR(input_nc=3, output_nc=2) 
    elif args.net_G == 'BIT':
        net = BASE_Transformer(input_nc=3, output_nc=2, token_len=4, resnet_stages_num=4, with_pos='learned', enc_depth=1, dec_depth=8)
    elif args.net_G == 'SEIFNet':
        net = SEIFNet(input_nc=3, output_nc=2)
    elif args.net_G == 'USSFC':
        net = USSFCNet(3, 1, ratio=0.5)
    elif args.net_G == 'GMDG':
        net = GMDG()
    elif args.net_G == 'HFANet':
        net = HFANet(input_channel=3, input_size=256)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % args.net_G)
    return init_net(net, init_type, init_gain, gpu_ids)


class ResNet(torch.nn.Module):
    def __init__(self, resnet_stages_num=5, backbone='resnet18'):
        super(ResNet, self).__init__()
        expand = 1
        if backbone == 'resnet18':
            self.resnet = models.resnet18(pretrained=True,
                                          replace_stride_with_dilation=[False,False,False])
        elif backbone == 'resnet34':
            self.resnet = models.resnet34(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
        elif backbone == 'resnet50':
            self.resnet = models.resnet50(pretrained=True,
                                          replace_stride_with_dilation=[False,True,True])
            expand = 4
        else:
            raise NotImplementedError
        self.relu = nn.ReLU()
        self.resnet_stages_num = resnet_stages_num

    def forward(self, x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x_1 = x

        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x) 
        x_2 = x

        x = self.resnet.layer2(x) 
        x_3 = x

        if self.resnet_stages_num > 3:
            x = self.resnet.layer3(x) 
            x_4 = x
            
        if self.resnet_stages_num == 5:
            x = self.resnet.layer4(x) 
            x_5 = x
        elif self.resnet_stages_num > 5:
            raise NotImplementedError
        return x_1, x_2, x_3, x_4, x_5

''' 
    Feature Extraction ResNet18
    torch.Size([8, 3, 256, 256])
    torch.Size([8, 64, 128, 128])
    torch.Size([18, 64, 64, 64])
    torch.Size([8, 128, 32, 32])
    torch.Size([8, 256, 16, 16])
    torch.Size([8, 512, 8, 8])
'''
# Gating Mechanism and Difference-Aware Guidance 
class GMDG(ResNet):
    def __init__(self, resnet_stages_num=5, backbone='resnet18'):
        super(GMDG, self).__init__(backbone=backbone, resnet_stages_num=resnet_stages_num)

        self.diffblock_1 = DiffEnhanceBlock(64)
        self.diffblock_2 = DiffEnhanceBlock(64)
        self.diffblock_3 = DiffEnhanceBlock(128)
        self.diffblock_4 = DiffEnhanceBlock(256)
        self.diffblock_5 = DiffEnhanceBlock(512)

        self.BiFPN = BiFPN(128,[64,64,128,256,512])
        
    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = super(GMDG, self).forward(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = super(GMDG, self).forward(x2)

        diff_1 = self.diffblock_1(x1_1, x2_1)
        diff_2 = self.diffblock_2(x1_2, x2_2)
        diff_3 = self.diffblock_3(x1_3, x2_3)
        diff_4 = self.diffblock_4(x1_4, x2_4)
        diff_5 = self.diffblock_5(x1_5, x2_5)

        x = [diff_1, diff_2, diff_3, diff_4, diff_5]
        out_1, out_2 = self.BiFPN(x)
        return out_1, out_2





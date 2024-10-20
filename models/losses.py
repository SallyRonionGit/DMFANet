import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    # [b,c,h,w] c is class num, not channel num
    target = target.long()
    # if target has been transformed into [b,1,h,w] before,we need to turn it back like(b,h,w)
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    # target.shape[1:] except the first dim other dims is consistant with target
    # input size of BCD always h,w and h == w, so if w of input is the same as target, h will be same too
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    # F.cross_entropy apply softmax in num_class dim of [b, num_class, h, w]
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)

############## Dice Loss ##############
class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1e-5
        input = torch.sigmoid(input)
        n, c, _, _ = input.shape
        # mask = np.zeros(shape=(targets.shape[2],targets.shape[3])) #256,256
        mask = target[0, 0, :, :]
        reversed_mask = mask
        reversed_mask = torch.where(reversed_mask == 1, 0, 1)
        i_m_reverse = reversed_mask
        mask_join = torch.zeros(size=(n, 2, mask.shape[0], mask.shape[1]))
        mask_join[:,0,...] = mask
        mask_join[:,1,...] = i_m_reverse
        mask_join = mask_join.to(device='cuda')

        num = mask_join.shape[0]
        input = input.reshape(num, -1)
        target = mask_join.reshape(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return dice
#######################################


############## Focal Loss #############
class FocalLoss(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):
        '''
        logits and label have same shape, and label data type is long
        args:
            logits: tensor of shape (N, ...)
            label: tensor of shape(N, ...)
        Usage is like this:
            >>> criteria = FocalLoss()
            >>> logits = torch.randn(8, 19, 384, 384)# nchw, float/half
            >>> lbs = torch.randint(0, 19, (8, 384, 384)) # nchw, int64_t
            >>> loss = criteria(logits, lbs)
        '''
        label = label.float()
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)

        ce_loss = self.crit(logits, label)

        focal_loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        if self.reduction == 'sum':
            focal_loss = focal_loss.sum()
        return focal_loss
############################################

########### Focal Loss + Dice Loss #########
class FocalLoss_with_dice(nn.Module):

    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',):
        super(FocalLoss_with_dice, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label):

        # compute loss
        # logits = logits.float() # use fp32 if logits is fp16

        n, c, _, _ = logits.shape
        # mask = label[0, 0, :, :]
        # reversed_mask = mask
        # reversed_mask = torch.where(reversed_mask == 1, 0, 1)
        # i_m_reverse = reversed_mask
        # mask_join = torch.zeros(size=(n, c, mask.shape[0], mask.shape[1]))
        # mask_join[:, 0, ...] = mask
        # mask_join[:, 1, ...] = i_m_reverse
        # label = mask_join.to(device='cuda')
        #
        # with torch.no_grad():
        #     alpha = torch.empty_like(logits).fill_(1 - self.alpha)
        #     alpha[label == 1] = self.alpha
        #
        # probs = torch.sigmoid(logits)
        # pt = torch.where(label == 1, probs, 1 - probs)

        if label.dim() == 4:
            label = torch.squeeze(label, dim=1)
        target = F.one_hot(label.long(), num_classes=2)
        target = target.permute(0, 3, 1, 2).float()

        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[target == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(target == 1, probs, 1 - probs)

        ce_loss = self.crit(logits, target)

        focal_loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        if self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        smooth=1e-5
        # logits=torch.sigmoid(logits)
        num=label.shape[0]
        probs=probs.reshape(num,-1)
        target=target.reshape(num,-1)
        # print(logits.type())
        # print(label.type())
        intersection=(probs*(target.float()))
        dice=(2.*intersection.sum(1)+smooth)/(probs.sum(1)+(target.float()).sum(1)+smooth)
        dice=1-dice.sum()/num
        loss = 0.5*focal_loss+0.5*dice
        return loss
###########################################
# bce + dice
def GMDG_Loss(inputs, targets):
    # print(inputs.shape, targets.shape)
    # bce = weight_binary_cross_entropy_loss(inputs,targets)

    n, c, _, _ = inputs.shape
    # mask = np.zeros(shape=(targets.shape[2],targets.shape[3])) #256,256
    if targets.dim() == 4:
        targets = torch.squeeze(targets, dim=1)
    targets = F.one_hot(targets.long(), num_classes=2)
    targets = targets.permute(0, 3, 1, 2).float()
    criterion = torch.nn.BCEWithLogitsLoss()  # 包含sigmoid
    bce = criterion(inputs, targets)

    smooth = 1e-5
    input = torch.sigmoid(inputs)
    num = targets.shape[0]
    input = input.reshape(num, -1)
    target = targets.reshape(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    # labuda = 0.75
    # loss = labuda*bce + (1-labuda)*dice
    loss = bce + dice

    # inter = (inputs * targets).sum()
    # eps = 1e-5
    # dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return loss

# bce + dice
def SEIFNet_Loss(inputs, targets):
    # print(inputs.shape, targets.shape)
    # bce = weight_binary_cross_entropy_loss(inputs,targets)

    n, c, _, _ = inputs.shape
    # mask = np.zeros(shape=(targets.shape[2],targets.shape[3])) #256,256
    if targets.dim() == 4:
        targets = torch.squeeze(targets, dim=1)
    targets = F.one_hot(targets.long(), num_classes=2)
    targets = targets.permute(0, 3, 1, 2).float()
    criterion = torch.nn.BCEWithLogitsLoss()  # 包含sigmoid
    bce = criterion(inputs, targets)

    smooth = 1e-5
    input = torch.sigmoid(inputs)
    num = targets.shape[0]
    input = input.reshape(num, -1)
    target = targets.reshape(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    labuda = 0.75
    loss = labuda*bce + (1-labuda)*dice
    # loss = bce + dice

    # inter = (inputs * targets).sum()
    # eps = 1e-5
    # dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return loss

# focal + dice
def SNUNet_Loss(inputs, targets):
    snunet_loss = FocalLoss_with_dice()
    loss = snunet_loss(inputs, targets)
    return loss

# bce + dice
def TFI_GR_Loss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice

# bce + dice
def A2Net_Loss(inputs, targets):
    # print(inputs.shape, targets.shape)
    bce = F.binary_cross_entropy(inputs, targets)
    inter = (inputs * targets).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return bce + 1 - dice

# ce
def DMINet_Loss(input, target, weight=None, reduction='mean',ignore_index=255):
    return cross_entropy(input, target, weight=None, reduction='mean',ignore_index=255)

# bce + dice
def IFNet_Loss(inputs,targets):
    # print(inputs.shape, targets.shape)
    # bce = weight_binary_cross_entropy_loss(inputs,targets)

    n, c, _, _ = inputs.shape
    # mask = np.zeros(shape=(targets.shape[2],targets.shape[3])) #256,256
    if targets.dim() == 4:
        targets = torch.squeeze(targets, dim=1)
    targets = F.one_hot(targets.long(), num_classes=2)
    targets = targets.permute(0, 3, 1, 2).float()
    criterion = torch.nn.BCEWithLogitsLoss()  # 包含sigmoid
    bce = criterion(inputs, targets)

    smooth = 1e-5
    input = torch.sigmoid(inputs)
    num = targets.shape[0]
    input = input.reshape(num, -1)
    target = targets.reshape(num, -1)
    intersection = (input * target)
    dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
    dice = 1 - dice.sum() / num
    # labuda = 0.75
    # loss = labuda*bce + (1-labuda)*dice
    loss = bce + dice

    # inter = (inputs * targets).sum()
    # eps = 1e-5
    # dice = (2 * inter + eps) / (inputs.sum() + targets.sum() + eps)
    # print(bce.item(), inter.item(), inputs.sum().item(), dice.item())
    return loss

def HFANet_Loss(input, target, weight=None, reduction='mean',ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    # [b,c,h,w] c is class num, not channel num
    target = target.long()
    # if target has been transformed into [b,1,h,w] before,we need to turn it back like(b,h,w)
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    # target.shape[1:] except the first dim other dims is consistant with target
    # input size of BCD always h,w and h == w, so if w of input is the same as target, h will be same too
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear',align_corners=True)
    # F.cross_entropy apply softmax in num_class dim of [b, num_class, h, w]
    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)









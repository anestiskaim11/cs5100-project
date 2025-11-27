import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np

EPS = 1e-6


def d_hinge_smooth(real_logits, fake_logits, smooth=0.1):
    """Discriminator hinge loss with label smoothing"""
    return F.relu(1.0 - smooth - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
def g_hinge(fake_logits):
    return -fake_logits.mean()
def feature_matching_loss(real_feats, fake_feats):
    return sum(F.l1_loss(a,b) for a,b in zip(real_feats, fake_feats))

def dice_loss(inputs, targets, eps=1e-6):
    """
    Dice Loss for multi-class segmentation.
    inputs: [B, C, H, W] (logits)
    targets: [B, H, W] (integer class indices)
    """
    num_classes = inputs.shape[1]
    targets_onehot = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
    probs = F.softmax(inputs, dim=1)
    
    intersection = torch.sum(probs * targets_onehot, dim=(0,2,3))
    union = torch.sum(probs + targets_onehot, dim=(0,2,3))
    dice = (2. * intersection + eps) / (union + eps)
    dice_loss_val = 1 - dice.mean()
    return dice_loss_val

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
        self.backup = None
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    @torch.no_grad()
    def apply(self, model):
        self.backup = {k: v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=False)
    @torch.no_grad()
    def restore(self, model):
        if self.backup is not None:
            model.load_state_dict(self.backup, strict=False)
            self.backup = None

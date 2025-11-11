import torch, torch.nn as nn, torch.nn.functional as F
from pytorch_msssim import ms_ssim
import numpy as np

EPS = 1e-6

def masked_l1(a,b,w):
    return ((a-b).abs()*w).sum()/(w.sum()+EPS)

def dice_loss(pred, target, w):
    pred = pred*w; target = target*w
    num = 2*(pred*target).sum((1,2,3))
    den = pred.sum((1,2,3)) + target.sum((1,2,3)) + EPS
    return (1 - num/den).mean()

def maxpool(x, k=3): return F.max_pool2d(x, k, 1, k//2)
def soft_erode(x): p1=-maxpool(-x,3); p2=-maxpool(-x,5); return torch.min(p1,p2)
def soft_dilate(x): return maxpool(x,3)
def soft_open(x): return soft_dilate(soft_erode(x))
def soft_skeletonize(img, it=10):
    img = torch.clamp(img, 0, 1); skel = torch.zeros_like(img)
    for _ in range(it):
        opened = soft_open(img)
        delta  = F.relu(img - opened)
        skel   = skel + F.relu(delta - skel*delta)
        img    = soft_erode(img)
    return torch.clamp(skel, 0, 1)
def cldice_loss(pred, target, w):
    pred = torch.clamp(pred,0,1)*w; target = torch.clamp(target,0,1)*w
    sp = soft_skeletonize(pred); st = soft_skeletonize(target)
    tprec = ((sp*target).sum((1,2,3))+EPS)/(sp.sum((1,2,3))+EPS)
    tsens = ((st*pred).sum((1,2,3))+EPS)/(st.sum((1,2,3))+EPS)
    return (1 - (2*tprec*tsens)/(tprec+tsens+EPS)).mean()

def dft_log_amp(x):
    X = torch.fft.rfft2(x, dim=(-2,-1))
    return torch.log(torch.clamp(X.abs(), min=1e-6))

def d_hinge(real_logits, fake_logits):
    return F.relu(1.0 - real_logits).mean() + F.relu(1.0 + fake_logits).mean()
def g_hinge(fake_logits):
    return -fake_logits.mean()


def focal_loss(inputs, targets, alpha=0.25, gamma=2.0, eps=1e-6):
    """
    Focal Loss for multi-class segmentation.
    inputs: [B, C, H, W] (logits)
    targets: [B, H, W] (integer class indices)
    """
    num_classes = inputs.shape[1]
    # Convert targets to one-hot
    targets_onehot = F.one_hot(targets, num_classes).permute(0,3,1,2).float()
    
    # Compute softmax over classes
    probs = F.softmax(inputs, dim=1).clamp(min=eps, max=1.0 - eps)
    
    ce_loss = -targets_onehot * torch.log(probs)
    focal_weight = alpha * (1 - probs) ** gamma
    loss = (focal_weight * ce_loss).sum(dim=1).mean()
    return loss


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

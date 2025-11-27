# gan.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from functools import partial

# ---------------------------
# Helpers
# ---------------------------
def conv3x3(in_ch, out_ch, stride=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1, bias=bias)

def conv1x1(in_ch, out_ch, stride=1, bias=True):
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=bias)

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, a=0.2)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
        nn.init.ones_(m.weight)
        if getattr(m, "bias", None) is not None:
            nn.init.zeros_(m.bias)

# ---------------------------
# Generator: ResNet34 encoder + decoder U-Net
# ---------------------------
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        # upsample then convs
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = conv3x3(in_ch + skip_ch, out_ch)
        self.act1 = nn.LeakyReLU(0.2, inplace=True)
        self.norm1 = nn.InstanceNorm2d(out_ch, affine=True)
        self.conv2 = conv3x3(out_ch, out_ch)
        self.act2 = nn.LeakyReLU(0.2, inplace=True)
        self.norm2 = nn.InstanceNorm2d(out_ch, affine=True)

    def forward(self, x, skip):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.act1(self.norm1(x))
        x = self.conv2(x)
        x = self.act2(self.norm2(x))
        return x

class GeneratorUNet(nn.Module):
    """
    Args:
        in_ch: number of input channels (e.g. 4)
        num_classes: output segmentation classes (logits)
        pretrained: bool, whether to use pretrained resnet34 weights
    Returns:
        y_hat_logits [B, num_classes, H, W],
        p_hat_logits [B, num_classes, H, W],
        feats: list of decoder feature maps (for optional uses)
    """
    def __init__(self, in_ch=4, num_classes=19, pretrained=True):
        super().__init__()
        # Load resnet34 and adapt first conv to in_ch
        resnet = models.resnet34(pretrained=pretrained)
        # take layers: conv1 -> bn1 -> relu -> maxpool -> layer1..layer4
        # replace conv1 if in_ch != 3
        if in_ch != 3:
            w = resnet.conv1.weight
            new_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # initialize new_conv
            nn.init.kaiming_normal_(new_conv.weight, a=0.2)
            if in_ch > 3:
                # copy first 3 channels and average for extra channels
                new_conv.weight.data[:, :3, :, :].copy_(w)
                if in_ch > 3:
                    for c in range(3, in_ch):
                        # copy channel 0 as a simple heuristic
                        new_conv.weight.data[:, c:c+1, :, :].copy_(w[:, :1, :, :])
            resnet.conv1 = new_conv
        self.encoder0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)  # /2
        self.pool = resnet.maxpool  # /4
        self.encoder1 = resnet.layer1  # /4
        self.encoder2 = resnet.layer2  # /8
        self.encoder3 = resnet.layer3  # /16
        self.encoder4 = resnet.layer4  # /32

        # Bridge
        self.bridge = nn.Sequential(
            conv3x3(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(512, affine=True),
            conv3x3(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(512, affine=True),
        )

        # Decoder: choose channels to keep model capacity reasonable
        self.dec4 = DecoderBlock(in_ch=512, skip_ch=256, out_ch=256)  # up -> /16
        self.dec3 = DecoderBlock(in_ch=256, skip_ch=128, out_ch=192)  # up -> /8
        self.dec2 = DecoderBlock(in_ch=192, skip_ch=64, out_ch=128)   # up -> /4
        self.dec1 = DecoderBlock(in_ch=128, skip_ch=64, out_ch=64)    # up -> /2
        self.dec0 = DecoderBlock(in_ch=64, skip_ch=0, out_ch=64)      # up -> /1

        # Output heads
        self.final_conv = nn.Sequential(
            conv3x3(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64, affine=True),
            conv1x1(64, num_classes)
        )
        # Auxiliary head (p_hat)
        self.aux_conv = nn.Sequential(
            conv3x3(64, 64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.InstanceNorm2d(64, affine=True),
            conv1x1(64, num_classes)
        )

        # init weights for decoder and heads
        self.apply(init_weights)

    def forward(self, x, m=None):
        """
        x: [B, in_ch, H, W] (your code passes f and m separately, but you call G(f, m))
        m: optional extra mask channel - we'll concatenate if provided
        """
        if m is not None:
            # ensure same spatial size; user passes m as [B,1,H,W]
            x_in = torch.cat([x, m], dim=1) if x.shape[1] == (self.encoder0[0].in_channels - 1) else x
            # NOTE: we've already accounted for in_ch outside; user constructs G with in_ch=4
        else:
            x_in = x

        # Encoder
        e0 = self.encoder0(x_in)       # /2
        p = self.pool(e0)              # /4
        e1 = self.encoder1(p)          # /4
        e2 = self.encoder2(e1)         # /8
        e3 = self.encoder3(e2)         # /16
        e4 = self.encoder4(e3)         # /32

        b = self.bridge(e4)

        d4 = self.dec4(b, e3)  # /16
        d3 = self.dec3(d4, e2) # /8
        d2 = self.dec2(d3, e1) # /4
        d1 = self.dec1(d2, e0) # /2
        d0 = self.dec0(d1, None) # /1

        y_hat_logits = self.final_conv(d0)   # [B, num_classes, H, W]
        p_hat_logits = self.aux_conv(d0)     # [B, num_classes, H, W]

        feats = [b, d4, d3, d2, d1, d0]
        return y_hat_logits, p_hat_logits, feats

# ---------------------------
# Patch Discriminator
# ---------------------------
class PatchDiscriminator(nn.Module):
    """
    Classic PatchGAN discriminator that returns (logits, feat_list).
    Input: in_ch channels (e.g. 3 + 1 + NUM_CLASSES)
    Output: logits [B,1,H_patch,W_patch], features: list of intermediate activations for feature-matching.
    """
    def __init__(self, in_ch=3, base_ch=64, n_layers=4, use_spectral=True):
        super().__init__()
        Conv = nn.utils.spectral_norm if use_spectral else (lambda x: x)

        layers = []
        feat_list = []
        # first layer (no norm)
        layers.append(Conv(nn.Conv2d(in_ch, base_ch, kernel_size=4, stride=2, padding=1)))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        # subsequent layers
        ch = base_ch
        for i in range(1, n_layers):
            nch = min(ch * 2, 512)
            layers.append(Conv(nn.Conv2d(ch, nch, kernel_size=4, stride=2 if i < n_layers - 1 else 1, padding=1)))
            layers.append(nn.InstanceNorm2d(nch, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            ch = nch

        # final conv -> single-channel patch output
        layers.append(Conv(nn.Conv2d(ch, 1, kernel_size=4, stride=1, padding=1)))

        self.model = nn.Sequential(*layers)
        self.apply(init_weights)

    def forward(self, x):
        """
        Returns:
            logits: [B,1,H',W']
            features: list of intermediate features for feature matching (list of tensors)
        """
        features = []
        out = x
        # iterate through modules collecting features after each conv block
        for module in self.model:
            out = module(out)
            # collect conv outputs (tensors) for feature matching - pick after convs and norms
            if isinstance(module, nn.Conv2d):
                features.append(out)
        # final logits
        logits = out
        return logits, features

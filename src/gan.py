import torch, torch.nn as nn, torch.nn.functional as F, timm
from config import NUM_CLASSES

def safe_create_convnext(pretrained=True):
    try:
        #return timm.create_model('convnext_tiny', pretrained=pretrained, features_only=True, out_indices=(0,1,2,3))
        return timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=pretrained, features_only=True, out_indices=(0,1,2,3))
    except Exception as e:
        print("ConvNeXt pretrained weights unavailable, falling back to random init.")
        return timm.create_model('convnext_base.fb_in22k_ft_in1k', pretrained=False, features_only=True, out_indices=(0,1,2,3))
        #return timm.create_model('convnext_tiny', pretrained=False, features_only=True, out_indices=(0,1,2,3))

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, k, 1, p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True),
        )
    def forward(self, x): return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch)
    def forward(self, x, skip):
        # Interpolate x to match the spatial size of skip
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        #x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class GeneratorUNet(nn.Module):
    def __init__(self, in_ch=4, num_classes=NUM_CLASSES): 
        super().__init__()
        self.backbone = safe_create_convnext(pretrained=True)
        chs = self.backbone.feature_info.channels()
        self.stem = ConvBlock(in_ch, 64)
        self.enc1 = ConvBlock(64, 96)

        self.up3 = UpBlock(chs[3], chs[2], 256)
        self.up2 = UpBlock(256, chs[1], 128)
        self.up1 = UpBlock(128, chs[0], 96)
        self.up0 = UpBlock(96, 96, 64)

        self.head_int = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(32, num_classes, 1), nn.Tanh())
        self.head_prob= nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.LeakyReLU(0.2, True),
                                      nn.Conv2d(32, num_classes, 1))

    def forward(self, f, m):
        # Shallow conditioning branch
        x0 = torch.cat([f, m], dim=1)
        x0 = self.stem(x0)
        x0 = self.enc1(x0)

        # ConvNeXt feature pyramid on RGB (repeat to 3ch)
        feats = self.backbone(f)
        s0, s1, s2, s3 = feats

        # Decode
        x = self.up3(s3, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        x = self.up0(x, x0)  # fuse shallow conditioned path at last stage

        y = self.head_int(x)             # [-1,1] synthetic OCT-A intensity
        plogits = self.head_prob(x)      # vessel logits
        p = torch.sigmoid(plogits)       # [0,1] vessel prob
        return y, p, plogits

class PatchDiscriminator(nn.Module):
    def __init__(self, in_ch, ndf=64):
        super().__init__()
        C=ndf
        self.b1 = nn.Sequential(nn.Conv2d(in_ch, C, 4, 2, 1), nn.LeakyReLU(0.2, True))
        self.b2 = nn.Sequential(nn.Conv2d(C, C*2, 4, 2, 1, bias=False), nn.BatchNorm2d(C*2), nn.LeakyReLU(0.2, True))
        self.b3 = nn.Sequential(nn.Conv2d(C*2, C*4, 4, 2, 1, bias=False), nn.BatchNorm2d(C*4), nn.LeakyReLU(0.2, True))
        self.b4 = nn.Sequential(nn.Conv2d(C*4, C*8, 4, 1, 1, bias=False), nn.BatchNorm2d(C*8), nn.LeakyReLU(0.2, True))
        self.head = nn.Conv2d(C*8, 1, 4, 1, 1)
    def forward(self, x):
        f1=self.b1(x); f2=self.b2(f1); f3=self.b3(f2); f4=self.b4(f3)
        out=self.head(f4); return out, [f1,f2,f3,f4]
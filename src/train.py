import torch, random, numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from gan import GeneratorUNet, PatchDiscriminator
from torch.optim.lr_scheduler import CosineAnnealingLR
from loss import EMA, masked_l1, d_hinge, g_hinge, dft_log_amp, soft_erode, dice_loss, cldice_loss
import torchvision.utils as vutils
from pytorch_msssim import ms_ssim
import numpy as np, os, time, cv2, random, torch
from dataloader import get_cityscapes_dataloader
from config import *
import argparse
import config

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def feature_matching_loss(real_feats, fake_feats):
    return sum(F.l1_loss(a,b) for a,b in zip(real_feats, fake_feats))

@torch.no_grad()
def evaluate(val_loader):
    G.eval(); ema.apply(G)
    ms_list, l1_list = [], []
    for batch in val_loader:
        f = batch["image"].to(device)
        y = batch["label"].to(device)
        m = batch["m"].to(device)
        y_hat, p_hat, _ = G(f, m)
        yh_01 = (y_hat+1)/2; y_01 = (y+1)/2
        ms = ms_ssim(yh_01, y_01, data_range=1.0, size_average=True)
        l1 = masked_l1(y_hat, y, m)
        ms_list.append(ms.item()); l1_list.append(l1.item())
    ema.restore(G); G.train()
    return {"ms_ssim": float(np.mean(ms_list)), "l1": float(np.mean(l1_list))}

def save_samples(batch, y_hat, p_hat, tag, max_n=4):
    f = batch["image"][:max_n].to(device)
    y = batch["label"][:max_n].to(device)
    #p = batch["prob"][:max_n].to(device)
    yh = y_hat[:max_n]; ph = p_hat[:max_n]

    # Convert 1-channel tensors to 3 channels by repeating
    y_3ch = y.repeat(1, 3, 1, 1)
    yh_3ch = yh.repeat(1, 3, 1, 1)
    #p_3ch = p.repeat(1, 3, 1, 1)
    ph_3ch = ph.repeat(1, 3, 1, 1)

    f_norm = (f - f.min())/(f.max()-f.min()+1e-6)
    panel = torch.cat([f_norm, (y_3ch+1)/2, (yh_3ch+1)/2, ph_3ch], dim=0)
    vutils.save_image(panel, f"{RUN_DIR}/samples/{tag}.png", nrow=max_n, normalize=False)

if __name__ == "__main__":


    if not os.path.exists(RUN_DIR):
        os.mkdir(RUN_DIR)
        os.mkdir(RUN_DIR+"/checkpoints")
        os.mkdir(RUN_DIR+"/inference")
        os.mkdir(RUN_DIR+"/report")
        os.mkdir(RUN_DIR+"/samples")
    
   
    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description="GAN training script")


    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="number of epochs")
    parser.add_argument("--lr", type=float, default=config.LR, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="batch size")
    parser.add_argument("--val_every", type=int, default=config.VAL_EVERY, help="validation frequency")

    args = parser.parse_args()


    EPOCHS     = args.epochs
    LR         = args.lr
    BATCH_SIZE = args.batch_size
    VAL_EVERY  = args.val_every

    device_str = "cuda" if torch.cuda.is_available() else "hip" if torch.version.hip else "cpu"
    device = torch.device(device_str)
    

    G = GeneratorUNet(in_ch=4, num_classes=NUM_CLASSES).to(device)
    D1 = PatchDiscriminator(in_ch=3+1+NUM_CLASSES).to(device)  
    D2 = PatchDiscriminator(in_ch=3+1+NUM_CLASSES).to(device)
    DY = PatchDiscriminator(in_ch=NUM_CLASSES).to(device)      # Y-only

    optG = Adam(G.parameters(),  lr=LR, weight_decay=1e-5, betas=(0.5, 0.999))
    optD1= Adam(D1.parameters(), lr=LR, weight_decay=1e-5, betas=(0.5, 0.999))
    optD2= Adam(D2.parameters(), lr=LR, weight_decay=1e-5, betas=(0.5, 0.999))
    optDY= Adam(DY.parameters(), lr=LR, weight_decay=1e-5, betas=(0.5, 0.999))

    schedG  = CosineAnnealingLR(optG,  T_max=EPOCHS, eta_min=1e-6)
    schedD1 = CosineAnnealingLR(optD1, T_max=EPOCHS, eta_min=1e-6)
    schedD2 = CosineAnnealingLR(optD2, T_max=EPOCHS, eta_min=1e-6)
    schedDY = CosineAnnealingLR(optDY, T_max=EPOCHS, eta_min=1e-6)

    ema = EMA(G, decay=0.999)
    scaler = torch.amp.GradScaler(enabled=True)

    print("Model params — G:", round(sum(p.numel() for p in G.parameters())/1e6,2), "M")

    best_score = -1e9
    start_epoch = 0

    _, train_loader = get_cityscapes_dataloader(mode='train')
    _, val_loader = get_cityscapes_dataloader(mode='val')


    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()
        G.train(); D1.train(); D2.train(); DY.train()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", leave=True)
        for i, batch in enumerate(pbar):
            f = batch["image"].to(device, non_blocking=True).float()   # [B,3,H,W]
            y = batch["label"].to(device, non_blocking=True).long()    # [B,1,H,W] or [B,H,W]
            m = batch["m"].to(device, non_blocking=True).float()       # [B,1,H,W]

            # ---- One-hot encode labels ----
            if y.ndim == 4:
                y = y.squeeze(1)  # [B,H,W]
            y_onehot = F.one_hot(y, num_classes=NUM_CLASSES)            # [B,H,W,NUM_CLASSES]
            y_onehot = y_onehot.permute(0,3,1,2).float()               # [B,NUM_CLASSES,H,W]

            # ----- D step -----
            with torch.amp.autocast(device_str):
                y_hat, p_hat, _ = G(f, m)  # y_hat: [B,NUM_CLASSES,H,W]

                # D1 full-res
                real1, rf1 = D1(torch.cat([f, m, y_onehot], dim=1))
                fake1, ff1 = D1(torch.cat([f, m, y_hat.detach()], dim=1))
                loss_d1 = d_hinge(real1, fake1)

                # D2 half-res
                f2  = F.interpolate(f, scale_factor=0.5, mode='bilinear', align_corners=False)
                m2  = F.interpolate(m, scale_factor=0.5, mode='bilinear', align_corners=False)
                y2  = F.interpolate(y_onehot, scale_factor=0.5, mode='bilinear', align_corners=False)
                yh2 = F.interpolate(y_hat.detach(), scale_factor=0.5, mode='bilinear', align_corners=False)
                real2, rf2 = D2(torch.cat([f2, m2, y2], dim=1))
                fake2, ff2 = D2(torch.cat([f2, m2, yh2], dim=1))
                loss_d2 = d_hinge(real2, fake2)

                # DY y-only
                realY_logits, _ = DY(y_onehot)
                fakeY_logits, _ = DY(y_hat.detach())
                loss_dy = d_hinge(realY_logits, fakeY_logits) * LAMBDA_GAN_Y

                loss_d = loss_d1 + loss_d2 + loss_dy

            optD1.zero_grad(set_to_none=True)
            optD2.zero_grad(set_to_none=True)
            optDY.zero_grad(set_to_none=True)
            scaler.scale(loss_d).backward()
            scaler.step(optD1)
            scaler.step(optD2)
            scaler.step(optDY)

            # ----- G step -----
            with torch.amp.autocast(device_str):
                y_hat, p_hat, plogits = G(f, m)

                # GAN losses
                fake1, ff1 = D1(torch.cat([f, m, y_hat], dim=1))
                fake2, ff2 = D2(torch.cat([
                    F.interpolate(f, scale_factor=0.5, mode='bilinear', align_corners=False),
                    F.interpolate(m, scale_factor=0.5, mode='bilinear', align_corners=False),
                    F.interpolate(y_hat, scale_factor=0.5, mode='bilinear', align_corners=False)
                ], dim=1))
                loss_g_gan = g_hinge(fake1) + g_hinge(fake2)

                # Y-only GAN
                fakeY_logits, _ = DY(y_hat)
                loss_g_y = g_hinge(fakeY_logits) * LAMBDA_GAN_Y

                # Feature matching
                real1, rf1 = D1(torch.cat([f, m, y_onehot], dim=1))
                real2, rf2 = D2(torch.cat([
                    F.interpolate(f, scale_factor=0.5, mode='bilinear', align_corners=False),
                    F.interpolate(m, scale_factor=0.5, mode='bilinear', align_corners=False),
                    F.interpolate(y_onehot, scale_factor=0.5, mode='bilinear', align_corners=False)
                ], dim=1))
                loss_fm = feature_matching_loss(rf1, ff1) + feature_matching_loss(rf2, ff2)

                # Reconstruction / masked L1
                l1 = masked_l1(y_hat, y_onehot, m)

                # Central frequency loss
                h,w = y_onehot.shape[-2:]
                cy0, cy1 = int(0.1*h), int(0.9*h)
                cx0, cx1 = int(0.1*w), int(0.9*w)
                cm = torch.zeros_like(m)
                cm[:,:,cy0:cy1,cx0:cx1] = m[:,:,cy0:cy1,cx0:cx1]
                l_freq = F.l1_loss(dft_log_amp(y_hat*cm), dft_log_amp(y_onehot*cm))

                # Rim loss
                er = soft_erode(m)
                rim = (m - er).clamp(0,1)
                l_rim = masked_l1(y_hat, y_onehot, rim)

                # Total G loss
                loss_g = (LAMBDA_GAN * loss_g_gan) + (LAMBDA_L1 * l1) + loss_g_y

            optG.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()
            scaler.step(optG)
            scaler.update()
            ema.update(G)

            pbar.set_postfix({
                "Loss_D": f"{loss_d.item():.4f}",
                "Loss_G": f"{loss_g.item():.4f}"
            })


        schedG.step()
        schedD1.step()
        schedD2.step()
        schedDY.step()


        # validation & checkpointing
        if (epoch % VAL_EVERY)==0:
            metrics = evaluate(val_loader)
            comp = metrics["ms_ssim"] - 0.5*metrics["l1"]
            dur = (time.time()-t0)/60
            #print(f"Epoch {epoch} done in {dur:.2f} min")
            print(f"[val] MS-SSIM={metrics['ms_ssim']:.4f}  L1={metrics['l1']:.4f}  Composite={comp:.4f}")
            

            # save samples using a validation batch
            with torch.no_grad():
                val_batch = next(iter(val_loader))
                f_val = val_batch["image"].to(device); m_val = val_batch["m"].to(device)
                ema.apply(G); # Apply EMA weights before inference
                y_hat_val, p_hat_val, _ = G(f_val, m_val)
                ema.restore(G); # Restore original weights
                save_samples(val_batch, y_hat_val, p_hat_val, tag=f"epoch_{epoch}", max_n=min(4,f_val.shape[0]))

            # checkpoints
            torch.save({
                "epoch": epoch,
                "G": G.state_dict(),
                "D1": D1.state_dict(),
                "D2": D2.state_dict(),
                "DY": DY.state_dict(),
                "optG": optG.state_dict(),
                "optD1": optD1.state_dict(),
                "optD2": optD2.state_dict(),
                "optDY": optDY.state_dict(),
                "ema": ema.shadow,
                "metrics": metrics,
            }, f"{RUN_DIR}/checkpoints/last.pt")

            if comp > best_score:
                best_score = comp
                torch.save({
                    "epoch": epoch,
                    "G": G.state_dict(),
                    "ema": ema.shadow,
                    "metrics": metrics,
                }, f"{RUN_DIR}/checkpoints/best.pt")
                #print(f"[BEST] New best composite={best_score:.4f}")
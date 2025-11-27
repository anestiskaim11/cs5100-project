import torch, random, numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from tqdm import tqdm
from gan import GeneratorUNet, PatchDiscriminator
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from loss import EMA, masked_l1, d_hinge, d_hinge_smooth, g_hinge, dft_log_amp, soft_erode, dice_loss, cldice_loss, focal_loss
import torchvision.utils as vutils
from pytorch_msssim import ms_ssim
from torchmetrics import JaccardIndex
import numpy as np, os, time, cv2, random, torch
from dataloader import get_cityscapes_dataloader
from config import *
import argparse
import config
import csv

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

def feature_matching_loss(real_feats, fake_feats):
    return sum(F.l1_loss(a,b) for a,b in zip(real_feats, fake_feats))



def evaluate(val_loader):
    G.eval()
    ema.apply(G)

    miou_metric = JaccardIndex(task='multiclass', num_classes=NUM_CLASSES).to(device)
    miou_metric.reset()

    with torch.no_grad():
        for batch in val_loader:
            f = batch["image"].to(device)
            y = batch["label"].to(device)   # keep original labels (may contain 255)
            m = batch["m"].to(device)

            # forward
            y_hat, p_hat, _ = G(f, m)           # [B, NUM_CLASSES, H, W]
            y_hat_cls = y_hat.argmax(dim=1)     # [B, H, W]

            # squeeze channel dim if present
            if y.ndim == 4:
                y = y.squeeze(1)                # [B, H, W]

            # Create mask of valid pixels: valid in [0, NUM_CLASSES-1] and not 255
            valid_mask = (y != 255) & (y >= 0) & (y < NUM_CLASSES)  # boolean [B,H,W]

            # If there are no valid pixels in this batch, skip update
            if valid_mask.any():
                # Flatten and apply mask
                preds_flat = y_hat_cls[valid_mask]  # 1D tensor of predicted labels for valid pixels
                targets_flat = y[valid_mask]        # 1D tensor of true labels for valid pixels

                # Ensure dtype is correct
                preds_flat = preds_flat.long()
                targets_flat = targets_flat.long()

                miou_metric.update(preds_flat, targets_flat)
            else:
                # optional: log skip
                # print("Skipping batch: no valid pixels for mIoU")
                continue

    # compute final metric (handle case of zero updates)
    try:
        miou = miou_metric.compute().item()
    except Exception:
        miou = float("nan")

    ema.restore(G)
    G.train()
    return {"mIoU": float(miou)}


def save_samples(batch, y_hat, p_hat, tag, max_n=4):
    f = batch["image"][:max_n].to(device)  # [B,3,H,W]
    y = batch["label"][:max_n].to(device)  # [B,H,W] or [B,1,H,W]

    # --- Convert y_hat and p_hat from one-hot/logits to class indices for visualization ---
    y_hat_cls = y_hat.argmax(dim=1, keepdim=True)  # [B,1,H,W]
    p_hat_cls = p_hat.argmax(dim=1, keepdim=True)  # [B,1,H,W]

    # --- Repeat to 3 channels for visualization ---
    y_3ch = y.repeat(1,3,1,1) if y.ndim==4 else y.unsqueeze(1).repeat(1,3,1,1)
    yh_3ch = y_hat_cls.repeat(1,3,1,1)
    ph_3ch = p_hat_cls.repeat(1,3,1,1)

    # --- Normalize input images ---
    f_norm = (f - f.min()) / (f.max() - f.min() + 1e-6)

    # --- Concatenate panels: input | GT | predicted | predicted-class ---
    panel = torch.cat([f_norm, y_3ch.float()/NUM_CLASSES, yh_3ch.float()/NUM_CLASSES, ph_3ch.float()/NUM_CLASSES], dim=0)

    # --- Save ---
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
    optD1= Adam(D1.parameters(), lr=LR*0.001, weight_decay=1e-5, betas=(0.5, 0.999))
    optD2= Adam(D2.parameters(), lr=LR*0.001, weight_decay=1e-5, betas=(0.5, 0.999))
    optDY= Adam(DY.parameters(), lr=LR*0.001, weight_decay=1e-5, betas=(0.5, 0.999))

    schedG  = ReduceLROnPlateau(optG, mode='max', factor=0.5, patience=5, min_lr=1e-6)
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

    train_G_losses = []
    train_D_losses = []
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

            # Map all invalid indices to the last valid class
            y = torch.clamp(y, max=NUM_CLASSES - 1)   # ensures labels ∈ [0,18]

            # Now safe to one-hot encode
            y_onehot = F.one_hot(y, num_classes=NUM_CLASSES)   # [B,H,W,19]
            y_onehot = y_onehot.permute(0, 3, 1, 2).float()    # [B,19,H,W]


            

            # ----- D step ----- (train D every 2 steps to balance with G)
            with torch.amp.autocast(device_str):
                y_hat_d, p_hat_d, _ = G(f, m)  # y_hat: [B,NUM_CLASSES,H,W]

                    # D1 full-res
                real1, rf1 = D1(torch.cat([f, m, y_onehot], dim=1))
                fake1, ff1 = D1(torch.cat([f, m, y_hat_d.detach()], dim=1))
                loss_d1 = d_hinge_smooth(real1, fake1, smooth=0.1)

                    # D2 half-res
                f2  = F.interpolate(f, scale_factor=0.5, mode='bilinear', align_corners=False)
                m2  = F.interpolate(m, scale_factor=0.5, mode='bilinear', align_corners=False)
                y2  = F.interpolate(y_onehot, scale_factor=0.5, mode='bilinear', align_corners=False)
                yh2 = F.interpolate(y_hat_d.detach(), scale_factor=0.5, mode='bilinear', align_corners=False)
                real2, rf2 = D2(torch.cat([f2, m2, y2], dim=1))
                fake2, ff2 = D2(torch.cat([f2, m2, yh2], dim=1))
                loss_d2 = d_hinge_smooth(real2, fake2, smooth=0.1)

                # DY y-only
                realY_logits, _ = DY(y_onehot)
                fakeY_logits, _ = DY(y_hat_d.detach())
                loss_dy = d_hinge_smooth(realY_logits, fakeY_logits, smooth=0.1) * LAMBDA_GAN_Y

                loss_d = loss_d1 + loss_d2 + loss_dy

                
                optD1.zero_grad(set_to_none=True)
                optD2.zero_grad(set_to_none=True)
                optDY.zero_grad(set_to_none=True)
                scaler.scale(loss_d).backward()

                scaler.unscale_(optD1)
                scaler.unscale_(optD2)
                scaler.unscale_(optDY)

                torch.nn.utils.clip_grad_norm_(D1.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(D2.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(DY.parameters(), max_norm=1.0)


                scaler.step(optD1)
                scaler.step(optD2)
                scaler.step(optDY)
            

            # ----- G step -----
            with torch.amp.autocast(device_str):
                # Compute y_hat for generator (reuse from D step if available, but need fresh forward pass for gradients)
                y_hat, p_hat, plogits = G(f, m)

                # GAN losses
                fake1, ff1 = D1(torch.cat([f, m, y_hat], dim=1))
                fake2, ff2 = D2(torch.cat([
                    F.interpolate(f, scale_factor=0.5, mode='bilinear', align_corners=False),
                    F.interpolate(m, scale_factor=0.5, mode='bilinear', align_corners=False),
                    F.interpolate(y_hat, scale_factor=0.5, mode='bilinear', align_corners=False)
                ], dim=1))
                loss_g_gan = g_hinge(fake1) + g_hinge(fake2)

                # Feature matching loss for stability
                # Get real features for feature matching
                _, rf1_real = D1(torch.cat([f, m, y_onehot], dim=1))
                f2_real = F.interpolate(f, scale_factor=0.5, mode='bilinear', align_corners=False)
                m2_real = F.interpolate(m, scale_factor=0.5, mode='bilinear', align_corners=False)
                y2_real = F.interpolate(y_onehot, scale_factor=0.5, mode='bilinear', align_corners=False)
                _, rf2_real = D2(torch.cat([f2_real, m2_real, y2_real], dim=1))
                loss_g_feat = feature_matching_loss(rf1_real, ff1) + feature_matching_loss(rf2_real, ff2)

                # Y-only GAN
                fakeY_logits, _ = DY(y_hat)
                loss_g_y = g_hinge(fakeY_logits) * LAMBDA_GAN_Y

                
                # Focal + Dice losses
                y_labels = y.squeeze(1) if y.ndim==4 else y
                ce_loss = F.cross_entropy(y_hat, y_labels)

                # Dice loss
                loss_dice = dice_loss(y_hat, y_labels)


                # Total G loss (with feature matching)
                loss_g = (LAMBDA_GAN * loss_g_gan) + loss_g_y + \
                (LAMBDA_FOCAL * ce_loss) + (LAMBDA_DICE * loss_dice) + \
                (0.1 * loss_g_feat)  # Feature matching weight


            optG.zero_grad(set_to_none=True)
            scaler.scale(loss_g).backward()

            scaler.unscale_(optG)
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)

            scaler.step(optG)
            scaler.update()
            ema.update(G)

            train_G_losses.append(loss_g.item())
            train_D_losses.append(loss_d.item())
            pbar.set_postfix({
                "Loss_D": f"{loss_d.item():.4f}",
                "Loss_G": f"{loss_g.item():.4f}"
            })


        schedD1.step()
        schedD2.step()
        schedDY.step()


        # validation & checkpointing
        if (epoch % VAL_EVERY)==0:
            metrics = evaluate(val_loader)
            comp = metrics["mIoU"]
            dur = (time.time()-t0)/60
            #print(f"Epoch {epoch} done in {dur:.2f} min")
            print(f"[val] mIoU={metrics['mIoU']:.4f}")
            
            # Step generator scheduler based on mIoU (ReduceLROnPlateau)
            schedG.step(metrics["mIoU"])
            

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

            csv_path = f"{RUN_DIR}/report/training_log.csv"
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode='a', newline='') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["epoch", "val_mIoU", "train_G_loss", "train_D_loss"])
                writer.writerow([
                    epoch,
                    metrics["mIoU"],
                    np.mean(train_G_losses),
                    np.mean(train_D_losses)
                ])
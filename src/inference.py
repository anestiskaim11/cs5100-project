import os
import torch
import cv2
import numpy as np
from tqdm import tqdm
from torchmetrics import JaccardIndex
from torchvision.utils import save_image

from gan import GeneratorUNet
from dataloader import get_cityscapes_dataloader
from config import *

# ---------------------------------------
# Utility: color map for Cityscapes
# ---------------------------------------
CITYSCAPES_COLORS = np.array([
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32)
], dtype=np.uint8)  # NUM_CLASSES = 19


def colorize(mask):
    """Convert H×W segmentation mask into a color image."""
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    valid = (mask >= 0) & (mask < NUM_CLASSES)
    color[valid] = CITYSCAPES_COLORS[mask[valid]]
    return color


# -------------------------------------------------------
#                    INFERENCE LOOP
# -------------------------------------------------------
def run_inference():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # Load generator
    G = GeneratorUNet(in_ch=4, num_classes=NUM_CLASSES).to(device)

    ckpt = torch.load(f"{RUN_DIR}/checkpoints/best.pt", map_location=device)
    G.load_state_dict(ckpt["G"])
    G.eval()

    # Apply EMA shadow weights if available
    if "ema" in ckpt:
        print("Using EMA weights for inference.")
        shadow = ckpt["ema"]
        for name, param in G.named_parameters():
            if name in shadow:
                param.data.copy_(shadow[name].data)

    # Create output folder
    out_dir = f"{RUN_DIR}/inference"
    os.makedirs(out_dir, exist_ok=True)

    _, test_loader = get_cityscapes_dataloader(mode="val")

    miou_metric = JaccardIndex(task="multiclass", num_classes=NUM_CLASSES).to(device)
    miou_metric.reset()

    print("Running inference...")
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader)):
            f = batch["image"].to(device)
            y = batch["label"].to(device)
            m = batch["m"].to(device)

            # Forward pass
            y_hat, _, _ = G(f, m)           # [B, NUM_CLASSES, H, W]
            y_pred = y_hat.argmax(1)        # [B,H,W]

            # VALID PIXELS: remove 255
            if y.ndim == 4:
                y = y.squeeze(1)

            valid_mask = (y != 255) & (y >= 0) & (y < NUM_CLASSES)

            
            if valid_mask.any():
                preds_flat = y_pred[valid_mask].long()
                targets_flat = y[valid_mask].long()
                miou_metric.update(preds_flat, targets_flat)

            # -------------------------------
            # Save visual outputs
            # -------------------------------
            for b in range(f.shape[0]):
                img = f[b].cpu().numpy().transpose(1,2,0)
                img = (img - img.min()) / (img.max() - img.min() + 1e-6)
                img = (img * 255).astype(np.uint8)

                pred_mask = y_pred[b].cpu().numpy().astype(np.uint8)
                pred_color = colorize(pred_mask)

                combined = np.concatenate([img, pred_color], axis=1)

                cv2.imwrite(f"{out_dir}/sample_{idx:05d}_{b}.png",
                            cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))

            

    # compute IoU
    
    miou = miou_metric.compute().item()
    

    print("\n==============================")
    print(f"  FINAL TEST mIoU = {miou:.4f}")
    print("==============================\n")


if __name__ == "__main__":
    run_inference()

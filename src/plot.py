import pandas as pd
import matplotlib.pyplot as plt
from config import *

# Load CSV file
df = pd.read_csv(f"{RUN_DIR}/report/training_log.csv")

# Some rows may restart epoch indexing (0–25 repeating)
# To visualize properly, we can add a global step index
df["step"] = range(len(df))

# Create figure
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot val_mIoU on left y-axis
color = 'tab:blue'
ax1.set_xlabel('Training step (approx. epochs)')
ax1.set_ylabel('Validation mIoU', color=color)
ax1.plot(df["step"], df["val_mIoU"], color=color, linewidth=2, label='Val mIoU')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(0, max(df["val_mIoU"]) * 1.1)

# Create second y-axis for losses
ax2 = ax1.twinx()
color_g = 'tab:red'
color_d = 'tab:green'
ax2.set_ylabel('Loss', color='black')
ax2.plot(df["step"], df["train_G_loss"], color=color_g, linestyle='--', label='G Loss')
ax2.plot(df["step"], df["train_D_loss"], color=color_d, linestyle='-.', label='D Loss')
ax2.tick_params(axis='y', labelcolor='black')

# Add title and grid
plt.title("Training Progress: mIoU and Losses over Epochs", fontsize=14)
ax1.grid(True, linestyle='--', alpha=0.6)

# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper right')

# Tight layout and save
plt.tight_layout()
plt.savefig(f"{RUN_DIR}/report/plot.png")

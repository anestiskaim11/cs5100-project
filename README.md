# CS5100 Project - GAN-based Semantic Segmentation

This project implements a GAN-based semantic segmentation model for the Cityscapes dataset, using a U-Net generator with ConvNeXt backbone and multiple patch discriminators.

## Installation

### Step 1: Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### Step 2: Install gdown

Install `gdown` to download datasets from Google Drive:

```bash
pip install gdown
```

### Step 3: Download Datasets

Download the Cityscapes dataset files:

```bash
gdown --id 1Ei7LWYR_LSzUUClNvUIEnZxfshaIA7fl --output leftImg8bit_trainvaltest.zip
gdown --id 19yEobWCfu_bTHHnzdrMQE-MN4t045Y85 --output gtFine_trainvaltest.zip
```

### Step 4: Extract Datasets

Extract the downloaded zip files:

```bash
unzip gtFine_trainvaltest.zip
unzip leftImg8bit_trainvaltest.zip
```

**Note:** On Windows, you may need to use a tool like 7-Zip or WinRAR, or use PowerShell:
```powershell
Expand-Archive gtFine_trainvaltest.zip
Expand-Archive leftImg8bit_trainvaltest.zip
```

## Usage

### Training

To train the model, run:

```bash
python src/train.py
```

You can also customize training parameters:

```bash
python src/train.py --epochs 100 --lr 1e-4 --batch_size 8 --val_every 1
```

Training outputs will be saved in the `run/` directory:
- `run/checkpoints/` - Model checkpoints (best.pt, last.pt)
- `run/samples/` - Sample predictions during training
- `run/report/` - Training logs (training_log.csv)

### Inference

To run inference on the validation set:

```bash
python src/inference.py
```

Inference results will be saved in `run/inference/` directory.

## Project Structure

```
cs5100-project/
├── src/
│   ├── __init__.py
│   ├── config.py          # Configuration parameters
│   ├── dataloader.py      # Cityscapes dataset loader
│   ├── gan.py             # Generator and Discriminator models
│   ├── loss.py            # Loss functions
│   ├── train.py           # Training script
│   ├── inference.py       # Inference script
│   └── plot.py            # Plotting utilities
├── leftImg8bit/           # Cityscapes images (after extraction)
├── gtFine/                # Cityscapes labels (after extraction)
├── run/                   # Training outputs
├── requirements.txt       # Python dependencies
└── README.md
```

## Configuration

Training parameters can be modified in `src/config.py`:

- `IMG_SIZE`: Input image size (default: 256)
- `BATCH_SIZE`: Batch size (default: 8)
- `NUM_CLASSES`: Number of segmentation classes (19 for Cityscapes)
- `EPOCHS`: Number of training epochs (default: 100)
- `LR`: Learning rate (default: 1e-4)
- `LAMBDA_GAN`, `LAMBDA_DICE`, `LAMBDA_CE`, `LAMBDA_FM`: Loss weights

## Requirements

- Python 3.8+
- PyTorch 2.8.0+ (with CUDA support recommended)
- See `requirements.txt` for full dependency list

## License

This project uses the Cityscapes dataset. Please refer to the license files in `leftImg8bit_trainvaltest/` and `gtFine_trainvaltest/` for dataset usage terms.

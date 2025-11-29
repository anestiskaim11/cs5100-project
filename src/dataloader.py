import os
import random
import torch
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from .config import *


# =============================================================================
# Cityscapes: 34 IDs → 19 trainIds mapping
# =============================================================================

CITYSCAPES_19_CLASSES = {
     0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
     7: 0,   # road
     8: 1,   # sidewalk
     9: 255, 10: 255,
     11: 2,  # building
     12: 3,  # wall
     13: 4,  # fence
     14: 255, 15: 255, 16: 255,
     17: 5,  # pole
     18: 255,
     19: 6,  # traffic light
     20: 7,  # traffic sign
     21: 8,  # vegetation
     22: 9,  # terrain
     23: 10, # sky
     24: 11, # person
     25: 12, # rider
     26: 13, # car
     27: 14, # truck
     28: 15, # bus
     29: 255, 30: 255,
     31: 16, # train
     32: 17, # motorcycle
     33: 18  # bicycle
}

# vectorized LUT for speed
_mapping_lut = np.zeros((256,), dtype=np.uint8) + 255
for k, v in CITYSCAPES_19_CLASSES.items():
    _mapping_lut[k] = v


def convert_34_to_19(label_tensor):
    """
    label_tensor: torch tensor (H,W) with values 0–33 (labelIds)
    returns mapped tensor (H,W) with values 0–18 or 255(ignore)
    """
    arr = label_tensor.numpy()
    mapped = _mapping_lut[arr]
    return torch.from_numpy(mapped)


# =============================================================================
# Helper Functions
# =============================================================================

def read_rgb(path, size=IMG_SIZE):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    assert im is not None, f"Failed to read image: {path}"
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    return im


def make_fov_mask(f_rgb):
    gray = cv2.cvtColor((f_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32) / 255.0
    m = (gray > 0.02).astype(np.float32)
    k = np.ones((5, 5), np.uint8)
    m = cv2.erode(m, k, iterations=1)
    return m


# =============================================================================
# Dataset
# =============================================================================

class CityscapesPairedDataset(Dataset):
    def __init__(self, paired_images_dict, image_transform=None, label_transform=None, augment=True):
        self.paired_images = paired_images_dict
        self.keys = list(paired_images_dict.keys())
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.augment = augment

        # augmentation params
        self.flip_prob = 0.5
        self.rotate_deg = 15
        self.translate = 0.05
        self.scale = (0.9, 1.1)

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        img_path = self.paired_images[key]['image']
        label_path = self.paired_images[key]['label']

        # read raw PIL images
        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)

        # generate FOV mask from resized cv2 image
        image_rgb = read_rgb(img_path)
        fov_mask = make_fov_mask(image_rgb)
        fov_mask = torch.from_numpy(fov_mask)[None]  # 1×H×W

        # original dims
        w, h = image.size

        # -------------------------
        # Data Augmentation
        # -------------------------
        if self.augment:
            # Horizontal flip
            if random.random() < self.flip_prob:
                image = TF.hflip(image)
                label = TF.hflip(label)
                fov_mask = TF.hflip(fov_mask)

            # Vertical flip
            if random.random() < self.flip_prob:
                image = TF.vflip(image)
                label = TF.vflip(label)
                fov_mask = TF.vflip(fov_mask)

            # Affine transforms
            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            max_dx = self.translate * w
            max_dy = self.translate * h
            translate = (random.uniform(-max_dx, max_dx),
                         random.uniform(-max_dy, max_dy))
            scale = random.uniform(*self.scale)

            image = TF.affine(image, angle, translate, scale, shear=0)
            label = TF.affine(label, angle, translate, scale, shear=0)
            fov_mask = TF.affine(fov_mask, angle, translate, scale, shear=0)

        # -------------------------
        # Transforms to tensors
        # -------------------------
        if self.image_transform:
            image = self.image_transform(image)

        if self.label_transform:
            label = self.label_transform(label).squeeze(0)  # H×W

        # Convert 34 IDs → 19 training IDs
        label = convert_34_to_19(label).long()

        return {
            'id': key,
            'image': image,
            'label': label,
            'm': fov_mask.float(),
            'path': img_path
        }


# =============================================================================
# Pairing Function
# =============================================================================

def get_cityscapes_pairs(root_left, root_label, split='train'):
    paired = {}
    no_match = 0

    split_left = os.path.join(root_left, split)
    split_label = os.path.join(root_label, split)

    for city in os.listdir(split_left):
        img_dir = os.path.join(split_left, city)
        lbl_dir = os.path.join(split_label, city)

        if not os.path.isdir(img_dir):
            continue

        for img_file in os.listdir(img_dir):
            if not img_file.endswith('_leftImg8bit.png'):
                continue

            base = img_file.replace('_leftImg8bit.png', '')
            label_file = base + '_gtFine_labelIds.png'

            img_path = os.path.join(img_dir, img_file)
            lbl_path = os.path.join(lbl_dir, label_file)

            if os.path.exists(lbl_path):
                paired[base] = {'image': img_path, 'label': lbl_path}
            else:
                no_match += 1

    print(f"[{split}] Found: {len(paired)} pairs | Missing labels: {no_match}")
    return paired


# =============================================================================
# Dataloader Factory
# =============================================================================

def get_cityscapes_dataloader(mode='train'):
    root_left = './leftImg8bit'
    root_label = './gtFine'

    paired = get_cityscapes_pairs(root_left, root_label, split=mode)

    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    label_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST),
        transforms.PILToTensor()
    ])

    dataset = CityscapesPairedDataset(
        paired_images_dict=paired,
        image_transform=img_transform,
        label_transform=label_transform,
        augment=(mode == 'train')
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(mode == 'train'),
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    return dataset, loader


# =============================================================================
# Debug Example
# =============================================================================

if __name__ == "__main__":
    for split in ['train', 'val']:
        print(f"\n--- Loading {split} ---")
        _, loader = get_cityscapes_dataloader(mode=split)
        for i, batch in enumerate(loader):
            print(f"Batch {i} ({split})")
            print("IDs:", batch['id'])
            print("Image:", batch['image'].shape)
            print("Label:", batch['label'].shape)
            print("Mask:", batch['m'].shape)
            if i >= 1:
                break

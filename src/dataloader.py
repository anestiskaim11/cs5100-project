import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
import cv2
from config import *




# --- HELPER FUNCTIONS --- #
def read_rgb(path, size=IMG_SIZE):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    assert im is not None, f"Failed to read {path}"
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
    im = im.astype(np.float32) / 255.0
    return im


def make_fov_mask(f_rgb):
    gray = cv2.cvtColor((f_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    m = (gray > 0.02).astype(np.float32)
    k = np.ones((5, 5), np.uint8)
    m = cv2.erode(m, k, iterations=1)
    return m


# --- DATASET CLASS --- #
class CityscapesPairedDataset(Dataset):
    def __init__(self, paired_images_dict, image_transform=None, label_transform=None, augment=True):
        self.paired_images = paired_images_dict
        self.keys = list(paired_images_dict.keys())
        self.image_transform = image_transform
        self.label_transform = label_transform
        self.augment = augment
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

        image = Image.open(img_path).convert('RGB')
        label = Image.open(label_path)
        w, h = image.size

        image_rgb = read_rgb(img_path)
        m = make_fov_mask(image_rgb)
        m = torch.from_numpy(m)[None]

        # --- Augmentation ---
        if self.augment:
            if random.random() < 0.5:
                image = TF.hflip(image)
                label = TF.hflip(label)
                m = TF.hflip(m)
            if random.random() < 0.5:
                image = TF.vflip(image)
                label = TF.vflip(label)
                m = TF.vflip(m)

            angle = random.uniform(-self.rotate_deg, self.rotate_deg)
            max_dx = self.translate * w
            max_dy = self.translate * h
            translations = (random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy))
            scale = random.uniform(self.scale[0], self.scale[1])

            image = TF.affine(image, angle=angle, translate=translations, scale=scale, shear=0)
            label = TF.affine(label, angle=angle, translate=translations, scale=scale, shear=0)
            m = TF.affine(m, angle=angle, translate=translations, scale=scale, shear=0)

        if self.image_transform:
            image = self.image_transform(image)
        if self.label_transform:
            label = self.label_transform(label)

        return {
            'id': key,
            'image': image,
            'label': label,
            'm': m.float()
        }


# --- PAIRING FUNCTION --- #
def get_cityscapes_pairs(root_left, root_label, split='train'):
    paired = {}
    no_match = 0

    split_left = os.path.join(root_left, split)
    split_label = os.path.join(root_label, split)

    for city in os.listdir(split_left):
        city_img_dir = os.path.join(split_left, city)
        city_lbl_dir = os.path.join(split_label, city)
        if not os.path.isdir(city_img_dir):
            continue

        for img_file in os.listdir(city_img_dir):
            if not img_file.endswith('_leftImg8bit.png'):
                continue
            base_name = img_file.replace('_leftImg8bit.png', '')
            label_file = base_name + '_gtFine_labelIds.png'
            label_path = os.path.join(city_lbl_dir, label_file)
            img_path = os.path.join(city_img_dir, img_file)
            if os.path.exists(label_path):
                paired[base_name] = {'image': img_path, 'label': label_path}
            else:
                no_match += 1

    print(f"[{split}] Total paired: {len(paired)}, No matches: {no_match}")
    return paired


# --- DATALOADER FACTORY --- #
def get_cityscapes_dataloader(mode='train'):
    root_left = 'cs5100-project/leftImg8bit'
    root_label = 'cs5100-project/gtFine'

    paired = get_cityscapes_pairs(root_left, root_label, split=mode)

    img_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    lbl_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE), interpolation=Image.NEAREST),
        transforms.PILToTensor()
    ])

    dataset = CityscapesPairedDataset(
        paired_images_dict=paired,
        image_transform=img_transform,
        label_transform=lbl_transform,
        augment=(mode == 'train')
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=(mode == 'train'),
        num_workers=NUM_WORKERS,
        pin_memory=True if torch.cuda.is_available() else False
    )
    return dataset, dataloader


# --- EXAMPLE --- #
if __name__ == "__main__":
    for split in ['train', 'val']:
        print(f"\n--- Loading {split} ---")
        _, dataloader = get_cityscapes_dataloader(mode=split)
        for i, batch in enumerate(dataloader):
            print(f"Batch {i} ({split}):")
            print("IDs:", batch['id'])
            print("Image shape:", batch['image'].shape)
            print("Label shape:", batch['label'].shape)
            if i >= 1:
                break

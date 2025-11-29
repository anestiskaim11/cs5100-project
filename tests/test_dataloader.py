import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import unittest
import numpy as np
import tempfile
import torchvision.transforms as transforms
from PIL import Image
from src.dataloader import (
    convert_34_to_19,
    read_rgb,
    make_fov_mask,
    CityscapesPairedDataset,
    get_cityscapes_pairs,
    CITYSCAPES_19_CLASSES
)
from src.config import IMG_SIZE


class TestDataloader(unittest.TestCase):
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        
    def test_convert_34_to_19(self):
        """Test label conversion from 34 to 19 classes"""
        # Create a tensor with some valid class IDs
        label_34 = torch.tensor([
            [7, 8, 11],   # road, sidewalk, building
            [26, 27, 28]  # car, truck, bus
        ])
        
        label_19 = convert_34_to_19(label_34)
        
        # Check that valid classes are mapped correctly
        self.assertEqual(label_19[0, 0].item(), CITYSCAPES_19_CLASSES[7])  # road -> 0
        self.assertEqual(label_19[0, 1].item(), CITYSCAPES_19_CLASSES[8])  # sidewalk -> 1
        self.assertEqual(label_19[0, 2].item(), CITYSCAPES_19_CLASSES[11])  # building -> 2
        
        # Check that invalid classes map to 255
        invalid_label = torch.tensor([[0, 1, 2]])  # invalid classes
        result = convert_34_to_19(invalid_label)
        self.assertTrue(torch.all(result == 255))
        
    def test_read_rgb(self):
        """Test RGB image reading"""
        # Create a test image
        test_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img_path = os.path.join(self.temp_dir, "test_img.png")
        Image.fromarray(test_img).save(img_path)
        
        img = read_rgb(img_path, size=IMG_SIZE)
        
        self.assertEqual(img.shape, (IMG_SIZE, IMG_SIZE, 3))
        self.assertTrue(np.all(img >= 0) and np.all(img <= 1))
        self.assertEqual(img.dtype, np.float32)
        
    def test_make_fov_mask(self):
        """Test FOV mask generation"""
        # Create a test RGB image
        f_rgb = np.random.rand(64, 64, 3).astype(np.float32)
        
        mask = make_fov_mask(f_rgb)
        
        self.assertEqual(mask.shape, (64, 64))
        self.assertTrue(np.all((mask == 0) | (mask == 1)))
        self.assertEqual(mask.dtype, np.float32)
        
    def test_cityscapes_dataset(self):
        """Test CityscapesPairedDataset"""
        # Create dummy image and label files
        img_dir = os.path.join(self.temp_dir, "images")
        label_dir = os.path.join(self.temp_dir, "labels")
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        
        # Create test files
        img_path = os.path.join(img_dir, "test_leftImg8bit.png")
        label_path = os.path.join(label_dir, "test_gtFine_labelIds.png")
        
        test_img = Image.new('RGB', (100, 100), color='red')
        test_label = Image.new('L', (100, 100), color=7)  # road class
        
        test_img.save(img_path)
        test_label.save(label_path)
        
        # Create paired dictionary
        paired = {
            "test": {
                "image": img_path,
                "label": label_path
            }
        }
        
        # Create transforms (required for the dataset to work)
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
        
        # Create dataset
        dataset = CityscapesPairedDataset(
            paired_images_dict=paired,
            image_transform=img_transform,
            label_transform=label_transform,
            augment=False
        )
        
        self.assertEqual(len(dataset), 1)
        
        # Get a sample
        sample = dataset[0]
        
        self.assertIn("image", sample)
        self.assertIn("label", sample)
        self.assertIn("m", sample)
        self.assertIn("path", sample)
        
        # Check shapes
        self.assertEqual(sample["image"].shape[0], 3)  # RGB channels
        self.assertEqual(len(sample["label"].shape), 2)  # H x W
        self.assertEqual(len(sample["m"].shape), 3)  # 1 x H x W


if __name__ == '__main__':
    unittest.main()


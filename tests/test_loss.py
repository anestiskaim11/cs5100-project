import torch
import unittest
import torch.nn as nn
from src.loss import (
    d_hinge_smooth, 
    g_hinge, 
    feature_matching_loss, 
    dice_loss,
    EMA
)
from src.config import NUM_CLASSES


class TestLoss(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 2
        self.num_classes = NUM_CLASSES
        self.height, self.width = 32, 32
        
    def test_d_hinge_smooth(self):
        """Test discriminator hinge loss with smoothing"""
        real_logits = torch.randn(self.batch_size, 1, 8, 8) + 1.0
        fake_logits = torch.randn(self.batch_size, 1, 8, 8) - 1.0
        
        loss = d_hinge_smooth(real_logits, fake_logits, smooth=0.1)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # scalar
        self.assertTrue(loss.item() >= 0)
        
    def test_g_hinge(self):
        """Test generator hinge loss"""
        fake_logits = torch.randn(self.batch_size, 1, 8, 8)
        
        loss = g_hinge(fake_logits)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # scalar
        
    def test_feature_matching_loss(self):
        """Test feature matching loss"""
        real_feats = [
            torch.randn(self.batch_size, 64, 32, 32),
            torch.randn(self.batch_size, 128, 16, 16),
            torch.randn(self.batch_size, 256, 8, 8)
        ]
        fake_feats = [
            torch.randn(self.batch_size, 64, 32, 32),
            torch.randn(self.batch_size, 128, 16, 16),
            torch.randn(self.batch_size, 256, 8, 8)
        ]
        
        loss = feature_matching_loss(real_feats, fake_feats)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # scalar
        self.assertTrue(loss.item() >= 0)
        
    def test_dice_loss(self):
        """Test dice loss for multi-class segmentation"""
        inputs = torch.randn(self.batch_size, self.num_classes, self.height, self.width)
        targets = torch.randint(0, self.num_classes, (self.batch_size, self.height, self.width))
        
        loss = dice_loss(inputs, targets)
        
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)  # scalar
        self.assertTrue(0 <= loss.item() <= 1)
        
    def test_ema_initialization(self):
        """Test EMA initialization"""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)
        
        self.assertEqual(len(ema.shadow), len(model.state_dict()))
        self.assertIsNone(ema.backup)
        
    def test_ema_update(self):
        """Test EMA update mechanism"""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)
        
        # Get initial shadow values
        initial_shadow = {k: v.clone() for k, v in ema.shadow.items()}
        
        # Update model parameters
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)
        
        # Update EMA
        ema.update(model)
        
        # Check that shadow values changed
        for key in ema.shadow:
            self.assertFalse(torch.equal(ema.shadow[key], initial_shadow[key]))
            
    def test_ema_apply_restore(self):
        """Test EMA apply and restore"""
        model = nn.Linear(10, 5)
        ema = EMA(model, decay=0.999)
        
        # Store original parameters
        original_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Update EMA
        with torch.no_grad():
            for param in model.parameters():
                param.add_(0.1)
        ema.update(model)
        
        # Apply EMA weights
        ema.apply(model)
        applied_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Restore original weights
        ema.restore(model)
        restored_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        # Check that original parameters were restored
        for key in original_params:
            self.assertTrue(torch.allclose(restored_params[key], original_params[key]))


if __name__ == '__main__':
    unittest.main()


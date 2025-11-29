import torch
import unittest
from src.gan import GeneratorUNet, PatchDiscriminator, ConvBlock, UpBlock
from src.config import NUM_CLASSES, IMG_SIZE


class TestGAN(unittest.TestCase):
    
    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = 2
        self.in_ch = 4
        self.num_classes = NUM_CLASSES
        
    def test_convblock(self):
        """Test ConvBlock forward pass"""
        block = ConvBlock(3, 64)
        x = torch.randn(2, 3, 32, 32)
        out = block(x)
        self.assertEqual(out.shape, (2, 64, 32, 32))
        
    def test_upblock(self):
        """Test UpBlock forward pass"""
        block = UpBlock(128, 64, 96)
        x = torch.randn(2, 128, 16, 16)
        skip = torch.randn(2, 64, 32, 32)
        out = block(x, skip)
        self.assertEqual(out.shape, (2, 96, 32, 32))
        
    def test_generator_forward(self):
        """Test GeneratorUNet forward pass"""
        G = GeneratorUNet(in_ch=self.in_ch, num_classes=self.num_classes)
        G = G.to(self.device)
        
        f = torch.randn(self.batch_size, 3, IMG_SIZE, IMG_SIZE).to(self.device)
        m = torch.randn(self.batch_size, 1, IMG_SIZE, IMG_SIZE).to(self.device)
        
        y, p, plogits = G(f, m)
        
        # Check output shapes
        self.assertEqual(y.shape, (self.batch_size, self.num_classes, IMG_SIZE, IMG_SIZE))
        self.assertEqual(p.shape, (self.batch_size, self.num_classes, IMG_SIZE, IMG_SIZE))
        self.assertEqual(plogits.shape, (self.batch_size, self.num_classes, IMG_SIZE, IMG_SIZE))
        
        # Check probability values are in [0, 1]
        self.assertTrue(torch.all(p >= 0) and torch.all(p <= 1))
        
    def test_discriminator_forward(self):
        """Test PatchDiscriminator forward pass"""
        in_ch = 3 + 1 + NUM_CLASSES  # image + mask + segmentation
        D = PatchDiscriminator(in_ch=in_ch)
        D = D.to(self.device)
        
        x = torch.randn(self.batch_size, in_ch, IMG_SIZE, IMG_SIZE).to(self.device)
        logits, features = D(x)
        
        # Check output shapes
        self.assertEqual(len(features), 4)
        self.assertTrue(logits.shape[0] == self.batch_size)
        self.assertTrue(logits.shape[1] == 1)
        
    def test_generator_discriminator_integration(self):
        """Test Generator and Discriminator work together"""
        G = GeneratorUNet(in_ch=self.in_ch, num_classes=self.num_classes).to(self.device)
        in_ch_d = 3 + 1 + NUM_CLASSES
        D = PatchDiscriminator(in_ch=in_ch_d).to(self.device)
        
        f = torch.randn(self.batch_size, 3, IMG_SIZE, IMG_SIZE).to(self.device)
        m = torch.randn(self.batch_size, 1, IMG_SIZE, IMG_SIZE).to(self.device)
        
        # Generator forward
        y_hat, p_hat, _ = G(f, m)
        
        # Discriminator forward on fake
        fake_input = torch.cat([f, m, y_hat], dim=1)
        fake_logits, _ = D(fake_input)
        
        self.assertTrue(fake_logits.shape[0] == self.batch_size)


if __name__ == '__main__':
    unittest.main()


import unittest
from src import config


class TestConfig(unittest.TestCase):
    
    def test_config_values(self):
        """Test that all config values are set and have valid types"""
        # Test RUN_DIR
        self.assertIsInstance(config.RUN_DIR, str)
        self.assertTrue(len(config.RUN_DIR) > 0)
        
        # Test IMG_SIZE
        self.assertIsInstance(config.IMG_SIZE, int)
        self.assertGreater(config.IMG_SIZE, 0)
        
        # Test BATCH_SIZE
        self.assertIsInstance(config.BATCH_SIZE, int)
        self.assertGreater(config.BATCH_SIZE, 0)
        
        # Test NUM_WORKERS
        self.assertIsInstance(config.NUM_WORKERS, int)
        self.assertGreaterEqual(config.NUM_WORKERS, 0)
        
        # Test NUM_CLASSES
        self.assertIsInstance(config.NUM_CLASSES, int)
        self.assertEqual(config.NUM_CLASSES, 19)
        
        # Test EPS
        self.assertIsInstance(config.EPS, float)
        self.assertGreater(config.EPS, 0)
        
        # Test SEED
        self.assertIsInstance(config.SEED, int)
        
        # Test loss weights
        self.assertIsInstance(config.LAMBDA_GAN, float)
        self.assertIsInstance(config.LAMBDA_DICE, float)
        self.assertIsInstance(config.LAMBDA_CE, float)
        self.assertIsInstance(config.LAMBDA_FM, float)
        self.assertIsInstance(config.LAMBDA_GAN_Y, float)
        self.assertGreaterEqual(config.LAMBDA_GAN, 0)
        self.assertGreaterEqual(config.LAMBDA_DICE, 0)
        self.assertGreaterEqual(config.LAMBDA_CE, 0)
        self.assertGreaterEqual(config.LAMBDA_FM, 0)
        self.assertGreaterEqual(config.LAMBDA_GAN_Y, 0)
        
        # Test training parameters
        self.assertIsInstance(config.EPOCHS, int)
        self.assertGreater(config.EPOCHS, 0)
        
        self.assertIsInstance(config.VAL_EVERY, int)
        self.assertGreater(config.VAL_EVERY, 0)
        
        self.assertIsInstance(config.LR, float)
        self.assertGreater(config.LR, 0)


if __name__ == '__main__':
    unittest.main()


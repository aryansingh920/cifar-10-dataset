import os
import yaml
from pathlib import Path


class TrainingConfig:
    """Configuration class for training parameters"""

    def __init__(self):
        # Paths
        self.BASE_DIR = Path(os.path.dirname(os.path.abspath(__file__))).parent
        self.DATA_DIR = self.BASE_DIR / "data"
        self.MODELS_DIR = self.BASE_DIR / "models"

        # Data paths
        self.TRAIN_LABELS_PATH = self.DATA_DIR / "CIFAR-10_Train_Labels.csv"
        self.TRAIN_IMAGES_DIR = self.DATA_DIR / "train"
        self.TEST_IMAGES_DIR = self.DATA_DIR / "test"
        self.AUGMENTED_TRAIN_DIR = self.DATA_DIR / "augmented_train"

        # Training parameters
        self.BATCH_SIZE = 32
        self.LEARNING_RATE = 0.001
        self.EPOCHS = 5
        self.NUM_CLASSES = 10
        self.VALIDATION_SPLIT = 0.2

        # Data augmentation
        self.AUGMENTATIONS_PER_IMAGE = 4
        self.SAMPLE_SIZE = 100

        # Model parameters
        self.DROPOUT_RATE = 0.3
        self.L1_FACTOR = 0.0001
        self.L2_FACTOR = 0.01

        # Early stopping
        self.PATIENCE = 5
        self.MIN_DELTA = 0.001

        # Create necessary directories
        self.create_directories()

    def create_directories(self):
        """Create necessary directories"""
        os.makedirs(self.DATA_DIR, exist_ok=True)
        os.makedirs(self.MODELS_DIR, exist_ok=True)
        os.makedirs(self.AUGMENTED_TRAIN_DIR, exist_ok=True)

    def save_config(self, path):
        """Save configuration to YAML file"""
        config_dict = {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(self).items()
                       if not k.startswith('_')}

        with open(path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load_config(cls, path):
        """Load configuration from YAML file"""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)

        config = cls()
        for k, v in config_dict.items():
            if hasattr(config, k):
                if isinstance(getattr(config, k), Path):
                    setattr(config, k, Path(v))
                else:
                    setattr(config, k, v)
        return config

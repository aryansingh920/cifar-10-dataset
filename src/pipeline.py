import torch
from preprocessing import CIFAR10PreProcessor, CSVGenerator
from trainer import CIFAR10Trainer
from config import TrainingConfig


class CIFAR10Pipeline:
    """Pipeline class for handling training workflow"""

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self.get_device()
        self.trainer = None

    @staticmethod
    def get_device():
        """Set up and return the appropriate device for training"""

        if torch.backends.mps.is_available():
            return torch.device('mps')
        elif torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def preprocess_data(self, augment_data=True):
        """Handle data preprocessing and augmentation"""
        processor = CIFAR10PreProcessor(
            self.config.TRAIN_LABELS_PATH,
            self.config.TRAIN_IMAGES_DIR,
            self.config.TEST_IMAGES_DIR
        )

        # Analyze and display data properties
        processor.load_and_display_images(num_samples=5, random_state=42)
        sizes, brightness, pixel_ranges = processor.analyze_image_properties(
            sample_size=self.config.SAMPLE_SIZE
        )

        if augment_data:
            # Create augmented dataset
            processor.create_and_save_augmented_images(
                augmented_dir=str(self.config.AUGMENTED_TRAIN_DIR),
                augmentations_per_image=self.config.AUGMENTATIONS_PER_IMAGE
            )

            # Generate CSV for augmented data
            generator = CSVGenerator()
            generator.script_to_generate_csv(
                fileName="train_labels",
                base_dir=str(self.config.AUGMENTED_TRAIN_DIR)
            )

    def train_model(self, hypertune=False):
        """Train the model"""
        print(f"Using device: {self.device}")

        self.trainer = CIFAR10Trainer(
            train_csv=str(self.config.DATA_DIR / "train_labels.csv"),
            train_dir=str(self.config.AUGMENTED_TRAIN_DIR),
            batch_size=self.config.BATCH_SIZE,
            learning_rate=self.config.LEARNING_RATE,
            num_classes=self.config.NUM_CLASSES,
            device=self.device
        )

        # Add hyperparameter tuning
        self.trainer.tune_hyperparameters(train_csv=str(self.config.DATA_DIR / "train_labels.csv"),
                                          train_dir=str(self.config.AUGMENTED_TRAIN_DIR)) if hypertune else None

        self.trainer.train(
            epochs=self.config.EPOCHS,
            # save_dir=str(self.config.MODELS_DIR)
            save_dir=str("hyper-models")
        )

    # In pipeline.py

    def train_transfer_model(self):
        self.trainer.train_with_transfer()
        metrics = self.trainer.compute_validation_metrics()
        return metrics


    def predict_image(self, image_path):
        """Make prediction for a single image"""
        if self.trainer is None:
            raise ValueError(
                "Model not trained. Please train the model first.")

        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        prediction = self.trainer.predict(image_path)
        print(f'Predicted class: {class_names[prediction]}')
        return class_names[prediction]

    def load_trained_model(self, model_path=None):
        """Load a trained model"""
        if model_path is None:
            model_path = self.config.MODELS_DIR / 'best_model.pt'

        if self.trainer is None:
            self.trainer = CIFAR10Trainer(
                train_csv=str(self.config.DATA_DIR / "train_labels.csv"),
                train_dir=str(self.config.AUGMENTED_TRAIN_DIR),
                batch_size=self.config.BATCH_SIZE,
                learning_rate=self.config.LEARNING_RATE,
                num_classes=self.config.NUM_CLASSES,
                device=self.device
            )

        self.trainer.load_model(str(model_path))

    def predict_test_directory(self, output_csv="predictions.csv"):
        """Predict classes for all test images and save results"""
        if self.trainer is None:
            raise ValueError(
                "Model not trained. Please load or train the model first.")

        predictions_df = self.trainer.predict_test_directory(
            test_dir=str(self.config.TEST_IMAGES_DIR),
            output_csv=output_csv
        )
        return predictions_df

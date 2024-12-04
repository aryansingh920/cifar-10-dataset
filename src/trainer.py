import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
from tqdm import tqdm
from modelRegularization import ModelRegularization, apply_regularization
from dataset import CIFAR10Dataset
from model import CIFAR10Model


class CIFAR10Trainer:
    def __init__(self, train_csv, train_dir, batch_size=32, learning_rate=0.001,
                 num_classes=10, device=None):
        """
        Initialize the trainer with regularization.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        # Set device
        if device is None:
            self.device = torch.device("mps" if torch.backends.mps.is_available()
                                       else "cuda" if torch.cuda.is_available()
                                       else "cpu")
        else:
            self.device = device
        print(f"Using device: {self.device}")

        # Create transforms with augmentation
        self.train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        # Setup datasets and dataloaders
        self.setup_data(train_csv, train_dir)

        # Initialize model with regularization
        self.model = apply_regularization(CIFAR10Model(
            num_classes=num_classes)).to(self.device)

        # Setup loss, optimizer and learning rate scheduler
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01  # L2 regularization
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        )

        # Initialize regularization handler
        self.regularization = ModelRegularization(
            model=self.model,
            patience=5,
            min_delta=0.001,
            l1_factor=0.0001,
            l2_factor=0.01
        )

    def setup_data(self, train_csv, train_dir, val_split=0.2):
        """Setup train and validation datasets and dataloaders"""
        dataset = CIFAR10Dataset(train_csv, train_dir, self.train_transform)

        train_size = int((1 - val_split) * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size]
        )

        val_dataset.dataset.transform = self.val_transform

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def train_epoch(self, epoch):
        """Train for one epoch with regularization"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch}')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)

            # Add regularization loss
            loss = self.criterion(outputs, labels)
            reg_loss = self.regularization.compute_regularization_loss()
            total_loss = loss + reg_loss

            total_loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': running_loss/len(self.train_loader),
                'acc': 100.*correct/total
            })

        return running_loss/len(self.train_loader), 100.*correct/total

    def validate(self):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        return running_loss/len(self.val_loader), 100.*correct/total

    def train(self, epochs=50, save_dir='models'):
        """Train the model with regularization and early stopping"""
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Update regularization history
            self.regularization.update_history(
                train_loss, val_loss, train_acc, val_acc)

            # Check for overfitting
            is_overfitting, gap = self.regularization.check_overfitting()
            if is_overfitting:
                print(
                    f"Warning: Possible overfitting detected (gap: {gap:.2f}%)")

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            print(f'Epoch {epoch}:')
            print(
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model(os.path.join(save_dir, 'best_model.pt'))
                print(f'Saved model with val_acc: {val_acc:.2f}%')

            # Early stopping check
            if self.regularization.should_stop(val_loss):
                print("Early stopping triggered!")
                break

            print('-' * 70)

        # Plot training history
        self.regularization.plot_training_history(
            save_path=os.path.join(save_dir, 'training_history.png')
        )

    def save_model(self, path):
        """Save the model"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)

    def load_model(self, path):
        """Load the model with weights_only=True"""
        checkpoint = torch.load(
            path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    def predict(self, image_path):
        """Predict class for a single image"""
        self.model.eval()
        transform = self.val_transform

        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(image)
            _, predicted = output.max(1)

        return predicted.item()

    def predict_test_directory(self, test_dir, output_csv="predictions.csv"):
        """Predict classes for all images in test directory and save to CSV."""
        idx_to_class = {
            0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'
        }

        predictions = []
        image_files = sorted([f for f in os.listdir(test_dir)
                              if f.endswith(('.png', '.jpg', '.jpeg'))])

        for image_file in tqdm(image_files, desc="Predicting test images"):
            image_path = os.path.join(test_dir, image_file)
            try:
                class_idx = self.predict(str(image_path))
                predictions.append({
                    'image_name': image_file,
                    'predicted_class': idx_to_class[class_idx]
                })
            except Exception as e:
                print(f"Error predicting {image_file}: {str(e)}")

        predictions_df = pd.DataFrame(predictions)
        predictions_df.to_csv(output_csv, index=False)
        print(f"\nPredictions saved to {output_csv}")
        return predictions_df

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

class ModelRegularization:
    def __init__(self, model: nn.Module, patience: int = 5, min_delta: float = 0.001,
                 l1_factor: float = 0.0, l2_factor: float = 0.01):
        """
        Initialize regularization and early stopping functionality.
        
        Args:
            model: The neural network model
            patience: Number of epochs to wait for improvement before early stopping
            min_delta: Minimum change in monitored quantity to qualify as an improvement
            l1_factor: L1 regularization factor
            l2_factor: L2 regularization factor
        """
        self.model = model
        self.patience = patience
        self.min_delta = min_delta
        self.l1_factor = l1_factor
        self.l2_factor = l2_factor
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        self.history: Dict[str, List[float]] = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
    def should_stop(self, val_loss: float) -> bool:
        """Check if training should stop based on validation loss."""
        if self.best_loss is None:
            self.best_loss = val_loss
            return False
            
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        return False
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """Compute L1 and L2 regularization losses."""
        l1_loss = torch.tensor(0., device=next(self.model.parameters()).device)
        l2_loss = torch.tensor(0., device=next(self.model.parameters()).device)
        
        for param in self.model.parameters():
            if param.requires_grad:
                l1_loss += torch.sum(torch.abs(param))
                l2_loss += torch.sum(param.pow(2))
        
        return self.l1_factor * l1_loss + self.l2_factor * l2_loss
    
    def update_history(self, train_loss: float, val_loss: float, 
                      train_acc: float, val_acc: float) -> None:
        """Update training history."""
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_acc'].append(val_acc)
    
    def plot_training_history(self, save_path: Optional[str] = None):
        """Plot training history."""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path)
        plt.show()
    
    def check_overfitting(self, threshold: float = 0.1) -> Tuple[bool, float]:
        """
        Check if model is overfitting based on train/val performance gap.
        
        Args:
            threshold: Maximum acceptable difference between train and val accuracy
            
        Returns:
            Tuple of (is_overfitting, gap)
        """
        if len(self.history['train_acc']) < 2:
            return False, 0.0
            
        train_acc = self.history['train_acc'][-1]
        val_acc = self.history['val_acc'][-1]
        gap = train_acc - val_acc
        
        return gap > threshold, gap

    def get_learning_curves(self) -> Dict[str, List[float]]:
        """Get learning curves data."""
        return self.history

def apply_regularization(model: nn.Module) -> nn.Module:
    """
    Apply regularization techniques to model architecture.
    
    Args:
        model: Original model
        
    Returns:
        Modified model with regularization
    """
    # Add dropout layers if not present
    def add_dropout(module):
        for name, child in module.named_children():
            if isinstance(child, (nn.Linear, nn.Conv2d)):
                module._modules[name] = nn.Sequential(
                    child,
                    nn.Dropout(p=0.3)
                )
            else:
                add_dropout(child)
    
    # Add batch normalization if not present
    def add_batchnorm(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Conv2d):
                module._modules[name] = nn.Sequential(
                    child,
                    nn.BatchNorm2d(child.out_channels)
                )
            elif isinstance(child, nn.Linear):
                module._modules[name] = nn.Sequential(
                    child,
                    nn.BatchNorm1d(child.out_features)
                )
            else:
                add_batchnorm(child)
    
    model_copy = type(model)()  # Create new instance of same model type
    model_copy.load_state_dict(model.state_dict())
    
    add_dropout(model_copy)
    add_batchnorm(model_copy)
    
    return model_copy

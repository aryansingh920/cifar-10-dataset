import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
from modelRegularization import ModelRegularization, apply_regularization


class CIFAR10Dataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Custom Dataset for CIFAR-10 images.
        
        Args:
            csv_file: Path to the csv file with annotations
            img_dir: Directory with all the images
            transform: Optional transform to be applied on a sample
        """
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.class_to_idx = {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(
            self.img_dir, self.data_frame.iloc[idx]['image_path'])
        image = Image.open(img_name).convert('RGB')
        label = self.class_to_idx[self.data_frame.iloc[idx]['label']]

        if self.transform:
            image = self.transform(image)

        return image, label

from torchvision.models import resnet18
import torch.nn as nn


class TransferModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TransferModel, self).__init__()
        self.resnet = resnet18(pretrained=True)

        # Freeze early layers
        for param in list(self.resnet.parameters())[:-2]:
            param.requires_grad = False

        # Replace final layer for CIFAR-10
        self.resnet.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.resnet(x)

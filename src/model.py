import torch.nn as nn


class CIFAR10Model(nn.Module):
    def __init__(self, num_classes=10, conv_filters=[32, 64, 128], dropout_rate=0.3, fc_units=512):
        super(CIFAR10Model, self).__init__()

        self.conv1 = self._make_conv_block(3, conv_filters[0], dropout_rate)
        self.conv2 = self._make_conv_block(
            conv_filters[0], conv_filters[1], dropout_rate)
        self.conv3 = self._make_conv_block(
            conv_filters[1], conv_filters[2], dropout_rate)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_filters[2] * 4 * 4, fc_units),
            nn.BatchNorm1d(fc_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_units, num_classes)
        )

    def _make_conv_block(self, in_channels, out_channels, dropout_rate):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x)
        return x

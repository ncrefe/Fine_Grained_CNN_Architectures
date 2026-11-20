import torch
import torch.nn as nn
import torch.nn.functional as F

# paste any helper classes here


class YourNet(nn.Module):
    def __init__(self, num_classes=25):
        super().__init__()

        # Block 1: Standart Conv
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: Standart Conv
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: Depthwise Separable (64 -> 96)
        self.dw3 = nn.Conv2d(64, 64, kernel_size=3, padding=1,
                             groups=64, bias=False)
        self.pw3 = nn.Conv2d(64, 96, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(96)

        # Block 4: Depthwise Separable (96 -> 128)
        self.dw4 = nn.Conv2d(96, 96, kernel_size=3, padding=1,
                             groups=96, bias=False)
        self.pw4 = nn.Conv2d(96, 128, kernel_size=1, bias=False)
        self.bn4 = nn.BatchNorm2d(128)

        # Block 5: Depthwise Separable (128 -> 160)
        self.dw5 = nn.Conv2d(128, 128, kernel_size=3, padding=1,
                             groups=128, bias=False)
        self.pw5 = nn.Conv2d(128, 160, kernel_size=1, bias=False)
        self.bn5 = nn.BatchNorm2d(160)

        #  Classifier head
        self.gap = nn.AdaptiveAvgPool2d((3, 3))
        self.drop = nn.Dropout(p=0.3)
        self.fc1 = nn.Linear(160 * 3 * 3, 256)
        self.bn_fc = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)  # 128 -> 64

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 64 -> 32

        # Block 3 (DWS)
        x = F.relu(self.bn3(self.pw3(self.dw3(x))))
        x = F.max_pool2d(x, 2)  # 32 -> 16

        # Block 4 (DWS)
        x = F.relu(self.bn4(self.pw4(self.dw4(x))))
        x = F.max_pool2d(x, 2)  # 16 -> 8

        # Block 5 (DWS) — No pooling, Spatial knowledge preserved
        x = F.relu(self.bn5(self.pw5(self.dw5(x))))  # 8 -> 8

        # Global pooling + classifier
        x = self.gap(x)  # 8 -> 3
        x = x.view(x.size(0), -1)  # 1440
        x = self.drop(x)
        x = F.relu(self.bn_fc(self.fc1(x)))
        x = self.drop(x)
        return self.fc2(x)

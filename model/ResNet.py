import torch.nn as nn
import torch


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()

        self.down_sample = False
        self.down_stride = 1

        if in_channel != out_channel:
            self.down_sample = True
            self.down_stride = 2
            self.conv_down = nn.Conv2d(in_channel, out_channel, 1, stride=2, padding=0)

        self.conv_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, out_channel, 3, stride=self.down_stride, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channel),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        identity = x
        x = self.conv_layer(x)

        if self.down_sample:
            identity = self.conv_down(identity)

        x += identity
        return x


class QResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(QResNet, self).__init__()

        # input 224x224x3
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )                               # 112x112x64
        self.pool = nn.MaxPool2d(2, 2)  # 56x56x64

        self.conv2_x = nn.Sequential(
            ResidualBlock(64, 64),      # 56x56x64
            ResidualBlock(64, 64),      # 56x56x64
            ResidualBlock(64, 64),      # 56x56x64
        )
        self.conv3_x = nn.Sequential(
            ResidualBlock(64, 128),     # 28x28x128
            ResidualBlock(128, 128),    # 28x28x128
            ResidualBlock(128, 128),    # 28x28x128
            ResidualBlock(128, 128),    # 28x28x128
        )
        self.conv4_x = nn.Sequential(
            ResidualBlock(128, 256),    # 14x14x256
            ResidualBlock(256, 256),    # 14x14x256
            ResidualBlock(256, 256),    # 14x14x256
            ResidualBlock(256, 256),    # 14x14x256
            ResidualBlock(256, 256),    # 14x14x256
            ResidualBlock(256, 256),    # 14x14x256
        )
        self.conv5_x = nn.Sequential(
            ResidualBlock(256, 512),    # 7x7x512
            ResidualBlock(512, 512),    # 7x7x512
            ResidualBlock(512, 512),    # 7x7x512
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2_x(x)
        x = self.conv3_x(x)
        x = self.conv4_x(x)
        x = self.conv5_x(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

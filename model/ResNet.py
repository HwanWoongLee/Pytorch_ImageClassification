import torch.nn as nn
import torch
import torchvision.models.resnet


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()

        self.down_sample = False
        self.down_stride = 1

        if in_channel != out_channel:
            self.down_sample = True
            self.down_stride = 2
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 1, stride=2, padding=0),
                nn.BatchNorm2d(out_channel),
            )

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=self.down_stride, padding=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv_layer(x)

        if self.down_sample:
            identity = self.conv_down(identity)

        x += identity
        x = self.relu(x)
        return x


class BottleneckBlock(nn.Module):
    def __init__(self, in_channel, mid_channel, out_channel, down=False):
        super(BottleneckBlock, self).__init__()

        self.down_stride = 1
        if down:
            self.down_stride = 2

        self.identity = True
        if in_channel != out_channel:
            self.identity = False

        self.conv_down = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, stride=self.down_stride, padding=0),
            nn.BatchNorm2d(out_channel),
        )

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channel, mid_channel, 1, stride=self.down_stride, padding=0),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),

            nn.Conv2d(mid_channel, mid_channel, 3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channel),
            nn.ReLU(),

            nn.Conv2d(mid_channel, out_channel, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_channel),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        x = self.conv_layer(x)

        if not self.identity:
            identity = self.conv_down(identity)
        x += identity
        x = self.relu(x)
        return x


class QResNet34(nn.Module):
    def __init__(self, num_classes=2):
        super(QResNet34, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2_x = nn.Sequential(
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
        )
        self.conv3_x = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
        )
        self.conv4_x = nn.Sequential(
            ResidualBlock(128, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
        )
        self.conv5_x = nn.Sequential(
            ResidualBlock(256, 512),
            ResidualBlock(512, 512),
            ResidualBlock(512, 512),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # input 224x224x3
        x = self.conv1(x)               # 112x112x64
        x = self.pool(x)                # 56x56x64
        x = self.conv2_x(x)             # 56x56x64
        x = self.conv3_x(x)             # 28x28x128
        x = self.conv4_x(x)             # 14x14x256
        x = self.conv5_x(x)             # 7x7x512
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class QResNet152(nn.Module):
    def __init__(self, num_classes=2):
        super(QResNet152, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool = nn.MaxPool2d(2, 2)

        self.conv2_x = nn.Sequential(
            BottleneckBlock(64, 64, 256),
            BottleneckBlock(256, 64, 256),
            BottleneckBlock(256, 64, 256),
        )
        self.conv3_x = nn.Sequential(BottleneckBlock(256, 128, 512, True))
        for i in range(7):
            self.conv3_x.add_module('conv3_{}'.format(i), BottleneckBlock(512, 128, 512))

        self.conv4_x = nn.Sequential(BottleneckBlock(512, 256, 1024, True))
        for i in range(35):
            self.conv4_x.add_module('conv4_{}'.format(i), BottleneckBlock(1024, 256, 1024))

        self.conv5_x = nn.Sequential(
            BottleneckBlock(1024, 512, 2048, True),
            BottleneckBlock(2048, 512, 2048),
            BottleneckBlock(2048, 512, 2048),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):       # input 224x224x3
        x = self.conv1(x)       # 112x112x64
        x = self.pool(x)        # 56x56x64
        x = self.conv2_x(x)     # 56x56x256
        x = self.conv3_x(x)     # 28x28x512
        x = self.conv4_x(x)     # 14x14x1024
        x = self.conv5_x(x)     # 7x7x2048
        x = self.avg_pool(x)    # 1x1x2048
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

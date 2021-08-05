import torch
import torch.nn as nn


def CBR2d(in_channels, out_channels, _kernal_size, _stride, _padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=_kernal_size, stride=_stride, padding=_padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


class QAlexNet(nn.Module):
    def __init__(self, num_classes=2):
        super(QAlexNet, self).__init__()

        # input 227 x 227 x 3
        self.feature = nn.Sequential(
            CBR2d(3, 96, 11, 4, 0),       # 55 x 55 x 96
            nn.MaxPool2d(3, 2, 0),        # 27 x 27 x 96
            CBR2d(96, 256, 5, 1, 2),      # 27 x 27 x 256
            nn.MaxPool2d(3, 2, 0),        # 13 x 13 x 256
            CBR2d(256, 384, 3, 1, 1),     # 13 x 13 x 384
            CBR2d(384, 384, 3, 1, 1),     # 13 x 13 x 384
            CBR2d(384, 256, 3, 1, 1),     # 13 X 13 X 256
            nn.MaxPool2d(3, 2, 0)         # 6 X 6 X 256
        )

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(6 * 6 * 256, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

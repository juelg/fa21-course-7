import torch
from torch import nn

def pad(f):
    return int((f - 1) / 2)


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # maybe add maxpooling
        self.net = nn.Sequential(
            # N x C x 256 x 256
            nn.Conv2d(3, 4, (5, 5),
                      stride=2, padding=pad(5)),
            nn.BatchNorm2d(num_features=4),
            nn.ReLU(inplace=True),
            # N x C x 128 x 128
            nn.Conv2d(4, 8, (5, 5), stride=2, padding=pad(5)),
            nn.BatchNorm2d(num_features=8),
            nn.ReLU(inplace=True),
            # N x 8 X 64 x 64
            nn.Conv2d(8, 16, (5, 5), stride=2, padding=pad(5)),
            nn.BatchNorm2d(num_features=16),
            nn.ReLU(inplace=True),
            # N x 16 X 32 x 32
            nn.Conv2d(16, 32, (5, 5), stride=2, padding=pad(5)),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(inplace=True),
            # N x 32 X 16 x 16
            nn.Conv2d(32, 64, (3, 3), stride=2, padding=pad(3)),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            # N x 64 X 8 x 8
            nn.Conv2d(64, 128, (3, 3), stride=2, padding=pad(3)),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(inplace=True),
            # N x 128 X 4 x 4
            nn.Conv2d(128, 256, (4, 4)),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(inplace=True),
            # N x 256 X 1 x 1
            nn.Conv2d(256, 1, (1, 1)),
            # N x 1 X 1 x 1
            nn.Flatten()
            # N x 1
        )
        # self.float()


    def forward(self, x):
        # print(x.shape)
        # print(x.dtype)
        n = self.net(x)
        # print(n.shape)
        # print(n.dtype)
        return n
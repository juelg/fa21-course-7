import torch
from torch import nn
from torch.nn.modules.activation import Tanh
import torchvision
from itertools import chain
import numpy as np
from torch.autograd import Variable

def pad(f):
    return int((f - 1) / 2)

  class StolenModel(torch.nn.Module):
      def __init__(self):
          super().__init__()
          # maybe add maxpooling
          self.net = nn.Sequential(
              # N x C x 256 x 256
              nn.Conv2d(3, 8, (3, 3), stride=1, padding= "valid"),
              nn.ReLU(),
              nn.MaxPool2d((2, 2), stride=None, padding= "valid"),
              nn.BatchNorm2d(num_features=8),
              nn.Conv2d(8, 16, (3, 3), stride=1, padding="valid"),
              nn.ReLU(),
              nn.MaxPool2d((2, 2), stride=None, padding= "valid"),
              nn.BatchNorm2d(num_features=16),
              nn.Conv2d(16, 64, (3, 3), stride=1, padding="valid"),
              nn.ReLU(),
              nn.MaxPool2d((2, 2), stride=None, padding= "valid"),
              nn.BatchNorm2d(num_features=64),
              nn.Conv2d(64, 128, (3, 3), stride=1, padding="valid"),
              nn.ReLU(),
              nn.MaxPool2d((2, 2), stride=None, padding= "valid"),
              nn.BatchNorm2d(num_features=128),
              nn.Conv2d(128, 256, (3, 3), stride=1, padding="valid"),
              nn.ReLU(),
              nn.MaxPool2d((2, 2), stride=None, padding= "valid"),
              nn.BatchNorm2d(num_features=256),
              nn.Flatten(),
              nn.Dropout2d(p=0.2)
              nn.Linear(256, 128),
              nn.ReLU(),
              nn.Linear(128, 64),
              nn.ReLU(),
              nn.Linear(64, 1),
              nn.tanh(),
        )

    def forward(self, x):
        n = self.net(x)
        return n





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
            nn.Flatten(),
            # N x 1
            # nn.Tanh()
        )


    def forward(self, x):
        n = self.net(x)
        return n

def get_flat_fts(in_size, fts):
    f = fts(Variable(torch.ones(1, *in_size)))
    return int(np.prod(f.size()[1:]))

class NvidiaModel(torch.nn.Module):
    def __init__(self, in_size=(3, 100, 320)):
        super().__init__()
        #320x160 -> 160, 320
        self.conv_net = nn.Sequential(
            # N x C x 72 x 320
            nn.Conv2d(3, 24, (5, 5),
                      stride=2, padding=pad(5)),
            nn.BatchNorm2d(num_features=24),
            nn.ELU(inplace=True),

            nn.Conv2d(24, 36, (5, 5), stride=2, padding=pad(5)),
            nn.BatchNorm2d(num_features=36),
            nn.ELU(inplace=True),

            nn.Conv2d(36, 48, (5, 5), stride=2, padding=pad(5)),
            nn.BatchNorm2d(num_features=48),
            nn.ELU(inplace=True),

            nn.Conv2d(48, 64, (5, 5), stride=2, padding=pad(5)),
            nn.BatchNorm2d(num_features=64),
            nn.ELU(inplace=True),

            nn.Conv2d(64, 64, (3, 3), padding=pad(3)),
            nn.BatchNorm2d(num_features=64),
            nn.ELU(inplace=True),

            nn.Flatten(),
            # dropout 0.2

        )
        # calculate what comes out
        out = get_flat_fts(in_size, self.conv_net)
        self.ful_net = nn.Sequential(
            nn.Linear(out, 100),
            nn.BatchNorm1d(num_features=100),
            nn.ELU(inplace=True),
            nn.Linear(100, 50),
            nn.BatchNorm1d(num_features=50),
            nn.ELU(inplace=True),
            nn.Linear(50, 10),
            nn.BatchNorm1d(num_features=10),
            nn.ELU(inplace=True),
            nn.Linear(10, 1),
            nn.Tanh()
        )
    def forward(self, x):
        n = self.conv_net(x)
        return self.ful_net(n)




class AdModel(torch.nn.Module):

    def __init__(self, num_features=500):
        super().__init__()
        self.model_conv = torchvision.models.resnet18(pretrained=True)
        # for param in self.model_conv.parameters():
        #     param.requires_grad = False
        num_ftrs = self.model_conv.fc.in_features
        self.model_conv.fc = nn.Linear(num_ftrs, num_features)

        self.reg = nn.Sequential(
            nn.BatchNorm1d(num_features=num_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_features, 1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.reg(self.model_conv(x))

    # def parameters(self):
    #     return chain(self.model_conv.fc.parameters(), self.reg.parameters())

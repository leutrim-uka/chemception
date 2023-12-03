import torch
import torch.nn as nn
from layers.units import Conv2d


class Stem(nn.Module):
    def __init__(self, in_channels):
        super(Stem, self).__init__()
        self.conv = Conv2d(in_channels, 32, 4, stride=2, padding=0, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class InceptionResNetA(nn.Module):
    def __init__(self, in_channels, scale=1.0):
        super(InceptionResNetA, self).__init__()
        self.scale = scale
        self.branch0 = Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False)
        self.branch1 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 32, 3, stride=1, padding=0, bias=False)
        )
        self.branch2 = nn.Sequential(
            Conv2d(in_channels, 32, 1, stride=1, padding=0, bias=False),
            Conv2d(32, 48, 3, stride=1, padding=0, bias=False),
            Conv2d(48, 64, 3, stride=1, padding=0, bias=False)
        )
        self.linear = Conv2d(64+32+32, in_channels, 1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b0 = self.branch0(x)
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        concat = torch.cat((b0, b1, b2), dim=1)
        linear = self.linear(concat)
        result = self.relu(x + linear)
        return result


class ReductionA(nn.Module):
    def __init__(self):
        super(ReductionA, self).__init__()

    def forward(self, x):
        pass


class InceptionResNetB(nn.Module):
    pass


class ReductionB(nn.Module):
    pass


class InceptionResNetC(nn.Module):
    pass


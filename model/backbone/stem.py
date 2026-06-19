import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False
        )
        self.gn = nn.GroupNorm(32, out_channels)
        self.act = nn.Mish()

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))

class E6TransStem(nn.Module):
    def __init__(self, in_channels=3, out_channels=512):
        super().__init__()
        self.block1 = ConvBlock(in_channels, 128, kernel_size=3, stride=2, padding=1)
        self.block2 = ConvBlock(128, 256, kernel_size=3, stride=2, padding=1)
        self.block3 = ConvBlock(256, out_channels, kernel_size=5, stride=4, padding=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return x
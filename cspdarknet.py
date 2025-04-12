import torch
import torch.nn as nn

# Basic convolution block
def conv_block(in_channels, out_channels, kernel_size, stride):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, kernel_size // 2, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.1)
    )

# CSP block
class CSPBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CSPBlock, self).__init__()
        hidden_channels = out_channels // 2
        self.part1 = nn.Sequential(
            conv_block(in_channels, hidden_channels, 1, 1),
            conv_block(hidden_channels, hidden_channels, 3, 1),
        )
        self.part2 = conv_block(in_channels, hidden_channels, 1, 1)
        self.final_conv = conv_block(hidden_channels * 2, out_channels, 1, 1)

    def forward(self, x):
        x1 = self.part1(x)
        x2 = self.part2(x)
        return self.final_conv(torch.cat([x1, x2], dim=1))

# Simplified CSPDarknet Backbone
class CSPDarknetBackbone(nn.Module):
    def __init__(self):
        super(CSPDarknetBackbone, self).__init__()
        self.layer1 = conv_block(3, 32, 3, 1)
        self.layer2 = conv_block(32, 64, 3, 2)
        self.layer3 = CSPBlock(64, 64)
        self.layer4 = conv_block(64, 128, 3, 2)
        self.layer5 = CSPBlock(128, 128)
        self.layer6 = conv_block(128, 256, 3, 2)
        self.layer7 = CSPBlock(256, 256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.out_dim = 256

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x

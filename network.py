import torch
from torch import nn

# Residual Block in Generative Network
# Small 3x3 kernels and 64 features map
class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super(ResidualBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x) # Elementwise Sum


# Upscaling Block in Generative Network
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels, scale_factor):
        super(UpsampleBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * (scale_factor ** 2),
                      kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.block(x)
    

# Generator Network
class Generator(nn.Module):
    def __init__(self, input_channel=3, num_blocks=16, upscale=4):
        super(Generator, self).__init__()

        # First layer: Conv + PReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channel, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        # Second layer: B Residual Blocks (Default 16)
        res_blocks = []
        for _ in range(num_blocks):
            res_blocks.append(ResidualBlock())
        self.residual_blocks = nn.Sequential(*res_blocks)

        # Third layer: Conv + BN after residual blocks
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Forth layer: Upscaling Block
        upsample_layers = []
        for _ in range(int(upscale/ 2)):  # 2 blocks for x4
            upsample_layers.append(UpsampleBlock(64, scale_factor=2))
        self.upsample= nn.Sequential(*upsample_layers)

        # Fifth layer: Conv
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        )

        # Sixth layer: Tanh Activation [-1 1]
        self.tanh = nn.Tanh()


    def forward(self, x):
        initial = self.conv1(x)
        res_out = self.residual_blocks(initial)
        out = initial + self.conv2(res_out)
        out = self.upsample(out)
        out = self.conv3(out)
        out = self.tanh(out)

        return out


# Discriminator Block in Discriminator Network
class DiscriminatorBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(DiscriminatorBlock, self).__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True) # Activation = 0.2
        ]
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)
    

# Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        # First layer: Conv + LeakyReLU
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )

        # Second layer: Discriminator Blocks from 64 to 512 kernels (Factor of 2)
        self.up_blocks = nn.Sequential(
            DiscriminatorBlock(64, 64, stride=2),
            DiscriminatorBlock(64, 128, stride=1),
            DiscriminatorBlock(128, 128, stride=2),
            DiscriminatorBlock(128, 256, stride=1),
            DiscriminatorBlock(256, 256, stride=2),
            DiscriminatorBlock(256, 512, stride=1),
            DiscriminatorBlock(512, 512, stride=2),
        )

        # Third layers: Two dense layer + Sigmoid activation function
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 6 * 6, 1024),  # depends on input size (96×96 HR)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.up_blocks(out)
        out = self.classifier(out)
        return out

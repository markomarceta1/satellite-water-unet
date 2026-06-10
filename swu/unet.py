import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive convolution layers with batch norm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool followed by double conv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv with skip connection."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        # after upsampling we will concatenate with the skip connection
        # the DoubleConv input channels should be (skip_channels + upsampled_channels)
        self.conv = DoubleConv(in_channels + out_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # x1: input from previous layer (to be upsampled)
        # x2: skip connection from encoder
        # Sanity check: conv_transpose expects `in_channels` equal to x1.shape[1]
        if x1.shape[1] != self.up.in_channels:
            raise RuntimeError(
                f"ConvTranspose2d expected input with {self.up.in_channels} channels, "
                f"but got {x1.shape[1]} channels. x1.shape={tuple(x1.shape)}, x2.shape={tuple(x2.shape)}, "
                f"up.weight.shape={tuple(self.up.weight.shape)}"
            )

        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = nn.functional.pad(
            x1,
            [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
        )

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """Final 1x1 convolution to map features to class logits."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UNet(nn.Module):
    """Simplified U-Net for binary segmentation."""

    def __init__(self, n_channels: int = 3, n_classes: int = 1, base_filters: int = 64):
        super().__init__()
        self.inc = DoubleConv(n_channels, base_filters)
        self.down1 = Down(base_filters, base_filters * 2)
        self.down2 = Down(base_filters * 2, base_filters * 4)
        self.down3 = Down(base_filters * 4, base_filters * 8)
        self.down4 = Down(base_filters * 8, base_filters * 8)
        # configure up blocks to match encoder channel sizes:
        # x5 (512) -> up1 (512->256), concat with x4 (512) => DoubleConv(512+256,256)
        self.up1 = Up(base_filters * 8, base_filters * 4)
        # x (256) -> up2 (256->128), concat with x3 (256) => DoubleConv(256+128,128)
        self.up2 = Up(base_filters * 4, base_filters * 2)
        # x (128) -> up3 (128->64), concat with x2 (128) => DoubleConv(128+64,64)
        self.up3 = Up(base_filters * 2, base_filters)
        # x (64) -> up4 (64->64), concat with x1 (64) => DoubleConv(64+64,64)
        self.up4 = Up(base_filters, base_filters)
        self.outc = OutConv(base_filters, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)

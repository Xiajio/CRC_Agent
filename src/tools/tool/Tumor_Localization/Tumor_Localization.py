"""
Tumor Localization U-Net Network
=================================

This is a complete, standalone U-Net implementation for tumor localization/segmentation tasks.
The network includes enhanced features like SE attention, residual connections, and ASPP.

Author: AI Assistant
Date: 2024
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""

    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """Enhanced Convolution Block with SE and residual connection"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.se = SEBlock(out_channels)
        self.residual = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.residual(x)
        out = self.double_conv(x)
        out = self.se(out)
        return out + identity


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.aspp = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, 1),
            nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6),
            nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12),
            nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18)
        ])
        self.gap = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True)
        )
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        size = x.size()[2:]
        gap = F.interpolate(self.gap(x), size=size, mode='bilinear', align_corners=True)
        outputs = [gap]
        outputs.extend([aspp(x) for aspp in self.aspp])
        out = torch.cat(outputs, dim=1)
        out = self.conv1(out)
        return self.relu(self.bn1(out))


class DownBlock(nn.Module):
    """Enhanced Downsampling Block"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.maxpool(x))


class UpBlock(nn.Module):
    """Enhanced Upsampling Block"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
            self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Enhanced U-Net for Tumor Localization/Segmentation

    Features:
    - SE (Squeeze-and-Excitation) attention blocks
    - Residual connections in convolution blocks
    - ASPP (Atrous Spatial Pyramid Pooling) in the bottleneck
    - Support for both bilinear and transposed convolution upsampling
    """

    def __init__(self, n_channels, n_classes, bilinear=False):
        """
        Initialize U-Net

        Args:
            n_channels (int): Number of input channels (e.g., 3 for RGB images)
            n_classes (int): Number of output classes (e.g., 2 for binary segmentation)
            bilinear (bool): Whether to use bilinear upsampling (True) or transposed conv (False)
        """
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Encoder path
        self.inc = ConvBlock(n_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024 // factor)

        # Bridge (bottleneck)
        self.bridge = ASPP(1024 // factor, 1024 // factor)

        # Decoder path
        self.up1 = UpBlock(1024, 512 // factor, bilinear)
        self.up2 = UpBlock(512, 256 // factor, bilinear)
        self.up3 = UpBlock(256, 128 // factor, bilinear)
        self.up4 = UpBlock(128, 64, bilinear)

        # Output layer
        self.outc = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
            nn.Dropout2d(0.1)
        )

    def forward(self, x):
        """
        Forward pass through the U-Net

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, n_channels, height, width)

        Returns:
            torch.Tensor: Output segmentation map of shape (batch_size, n_classes, height, width)
        """
        # Encoder path with skip connections
        x1 = self.inc(x)      # (batch, 64, H, W)
        x2 = self.down1(x1)   # (batch, 128, H/2, W/2)
        x3 = self.down2(x2)   # (batch, 256, H/4, W/4)
        x4 = self.down3(x3)   # (batch, 512, H/8, W/8)
        x5 = self.down4(x4)   # (batch, 1024//factor, H/16, W/16)

        # Bridge
        x5 = self.bridge(x5)  # (batch, 1024//factor, H/16, W/16)

        # Decoder path with skip connections
        x = self.up1(x5, x4)  # (batch, 512//factor, H/8, W/8)
        x = self.up2(x, x3)   # (batch, 256//factor, H/4, W/4)
        x = self.up3(x, x2)   # (batch, 128//factor, H/2, W/2)
        x = self.up4(x, x1)   # (batch, 64, H, W)

        # Output
        logits = self.outc(x)  # (batch, n_classes, H, W)

        return logits

    def get_parameter_count(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_tumor_localization_model(n_channels=3, n_classes=2, bilinear=False):
    """
    Factory function to create a tumor localization U-Net model

    Args:
        n_channels (int): Number of input channels (default: 3 for RGB)
        n_classes (int): Number of output classes (default: 2 for binary segmentation)
        bilinear (bool): Whether to use bilinear upsampling (default: False)

    Returns:
        UNet: Configured U-Net model
    """
    model = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)

    # Print model information
    print("Tumor Localization U-Net Created")
    print(f"Input channels: {n_channels}")
    print(f"Output classes: {n_classes}")
    print(f"Upsampling: {'Bilinear' if bilinear else 'Transposed Conv'}")
    print(f"Total parameters: {model.get_parameter_count():,}")
    print("=" * 50)

    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_tumor_localization_model(n_channels=3, n_classes=2)

    # Test with dummy input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Create dummy input (batch_size=1, channels=3, height=256, width=256)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)

    print("Testing model with dummy input...")
    with torch.no_grad():
        output = model(dummy_input)
        print(f"Input shape: {dummy_input.shape}")
        print(f"Output shape: {output.shape}")

    print("Tumor Localization U-Net is ready for use!")
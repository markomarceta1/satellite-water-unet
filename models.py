"""Top-level model wrapper for the project.

This file exposes the primary `UNet` model so notebooks and external users
can import `from models import UNet` without referring to the package path.
"""
from swu.unet import UNet

__all__ = ["UNet"]

def build_unet(n_channels: int = 3, n_classes: int = 1, base_filters: int = 64) -> UNet:
    """Convenience constructor for the U-Net model."""
    return UNet(n_channels=n_channels, n_classes=n_classes, base_filters=base_filters)

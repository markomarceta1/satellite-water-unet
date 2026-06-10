# Helper functions for creating synthetic cloud occlusions on image tensors.
# These functions are intentionally simple and designed to make the
# masking behavior easy to understand and modify.
import random
from functools import partial
from typing import Callable, Optional

from torch import Tensor


def add_fake_cloud(
    image: Tensor,
    patch_count: int = 2,
    min_patch_size: int = 30,
    max_patch_size: int = 80,
) -> Tensor:
    """Apply one or more synthetic cloud patches to an image tensor.

    The input tensor is expected to have shape (C, H, W).
    Each patch is created by zeroing out a random rectangular region.
    """
    if image.ndim != 3:
        raise ValueError("Expected image tensor shape (C, H, W)")

    # Validate the patch configuration.
    if patch_count < 1:
        raise ValueError("patch_count must be at least 1")
    if max_patch_size < min_patch_size:
        raise ValueError("max_patch_size must be greater than or equal to min_patch_size")

    _, height, width = image.shape
    if height < 2 or width < 2:
        # If the image is too small, skip masking entirely.
        return image

    for _ in range(patch_count):
        # Compute the largest patch size that fits within half the image
        # so cloud blocks remain reasonably sized relative to the image.
        max_w = min(max_patch_size, width // 2)
        max_h = min(max_patch_size, height // 2)
        min_w = min(min_patch_size, max_w)
        min_h = min(min_patch_size, max_h)

        if min_w < 1 or min_h < 1:
            break

        patch_width = random.randint(min_w, max_w)
        patch_height = random.randint(min_h, max_h)

        # Choose a random location for the patch inside the image bounds.
        x = random.randint(0, width - patch_width)
        y = random.randint(0, height - patch_height)

        # Zero out the selected region to simulate a cloud.
        image[:, y : y + patch_height, x : x + patch_width] = 0

    return image


def create_masked_image(
    image: Tensor,
    cloud_transform: Optional[Callable[[Tensor], Tensor]] = None,
) -> Tensor:
    """Return a cloud-masked copy of the input image tensor.

    This helper clones the input tensor before applying the cloud transform,
    so the original tensor is preserved.
    """
    cloud_transform = cloud_transform or partial(
        add_fake_cloud,
        patch_count=2,
        min_patch_size=30,
        max_patch_size=80,
    )
    return cloud_transform(image.clone())

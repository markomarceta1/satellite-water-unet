# Dataset loading utilities for satellite water segmentation.
# This module defines a PyTorch Dataset that returns original images,
# synthetic cloud-masked images, and ground truth water masks.
from functools import partial
from pathlib import Path
from typing import Callable, Optional

from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

from .masking import add_fake_cloud


IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_EXTENSIONS


def _sorted_image_paths(directory: Path):
    return sorted(p for p in directory.iterdir() if is_image_file(p))


class WaterDataset(Dataset):
    """PyTorch dataset for satellite imagery with synthetic cloud masking."""

    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
        cloud_transform: Optional[Callable[[Tensor], Tensor]] = None,
        cloud_patch_count: int = 2,
        cloud_min_patch_size: int = 30,
        cloud_max_patch_size: int = 80,
    ):
        # Convert the provided directory strings into Path objects
        # so we can use pathlib for robust file path operations.
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)

        # Validate that both directories exist before trying to scan files.
        if not self.image_dir.is_dir():
            raise ValueError(f"Image directory does not exist: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise ValueError(f"Mask directory does not exist: {self.mask_dir}")

        # Find all supported image files inside each directory.
        images = _sorted_image_paths(self.image_dir)
        masks = _sorted_image_paths(self.mask_dir)

        if not images:
            raise ValueError(f"No image files found in {self.image_dir}")
        if not masks:
            raise ValueError(f"No mask files found in {self.mask_dir}")

        # Build a lookup table of masks keyed by filename stem.
        # This ensures we pair each image with the correct mask file.
        mask_lookup = {mask.stem: mask for mask in masks}
        self.images = []
        self.masks = []

        for image_path in images:
            # Use the base filename (stem) to find the corresponding mask.
            mask_path = mask_lookup.get(image_path.stem)
            if mask_path is None:
                raise ValueError(
                    f"No matching mask found for image: {image_path.name}"
                )
            self.images.append(image_path)
            self.masks.append(mask_path)

        # Detect any masks that don't have a matching image.
        if len(self.images) != len(masks):
            unmatched_masks = [mask.name for mask in masks if mask.stem not in {img.stem for img in self.images}]
            raise ValueError(
                "Extra masks were found without matching images: "
                + ", ".join(unmatched_masks)
            )

        # Set up image transformations. Use the provided transforms if available,
        # otherwise resize images to a smaller fixed size and convert to tensors.
        default_img_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        default_mask_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])

        self.transform = transform or default_img_transform
        self.mask_transform = mask_transform or default_mask_transform

        # Use an external cloud transform helper to generate synthetic occlusions.
        # By default, create multiple cloud patches with larger default sizes.
        self.cloud_transform = cloud_transform or partial(
            add_fake_cloud,
            patch_count=cloud_patch_count,
            min_patch_size=cloud_min_patch_size,
            max_patch_size=cloud_max_patch_size,
        )

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        # Standard PyTorch Dataset indexing guard.
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")

        image_path = self.images[idx]
        mask_path = self.masks[idx]

        # Load the image and mask from disk. Convert the image to RGB and the mask
        # to a single channel grayscale image, which is typical for binary masks.
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # Apply the configured transforms to convert PIL images into tensors.
        image_tensor = self.transform(image)
        mask_tensor = self.mask_transform(mask)

        # Generate a synthetic cloud-covered version of the image.
        masked_image = self.cloud_transform(image_tensor.clone())

        # Return the original image, the cloud-masked image, and the target mask.
        return image_tensor, masked_image, mask_tensor

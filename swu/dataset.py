import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random


class WaterDataset(Dataset):

    def __init__(self, image_dir, mask_dir):

        self.image_dir = image_dir
        self.mask_dir = mask_dir

        def is_image_file(filename):
            return filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))

        self.images = sorted([f for f in os.listdir(image_dir) if is_image_file(f)])
        self.masks = sorted([f for f in os.listdir(mask_dir) if is_image_file(f)])

        self.transform = transforms.ToTensor()

    def add_fake_cloud(self, image):

        _, h, w = image.shape

        x = random.randint(0, w // 2)
        y = random.randint(0, h // 2)

        cloud_w = random.randint(20, 50)
        cloud_h = random.randint(20, 50)

        image[:, y:y+cloud_h, x:x+cloud_w] = 0

        return image

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.masks[idx])

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        masked_image = self.add_fake_cloud(image.clone())

        return image, masked_image, mask
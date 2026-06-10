import sys
from pathlib import Path

ROOT = str(Path(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader, random_split

from swu.dataset import WaterDataset
from swu.unet import UNet
from swu.metrics import iou_score, dice_score


def find_checkpoint():
    out = Path("outputs")
    candidates = [out / "unet_best.pth", out / "unet_epoch15.pth", out / "unet_epoch10.pth", out / "unet_epoch5.pth"]
    for c in candidates:
        if c.exists():
            return c
    return None


def main():
    ds = WaterDataset(image_dir="data/images", mask_dir="data/masks")
    n = len(ds)
    test_size = max(1, int(n * 0.15))
    val_size = max(1, int(n * 0.15))
    train_size = n - val_size - test_size

    train_ds, val_ds, test_ds = random_split(
        ds, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

    ckpt = find_checkpoint()
    if ckpt is None:
        print("No checkpoint found in outputs/ — aborting.")
        return

    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    total_iou = 0.0
    total_dice = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in test_loader:
            _, masked_images, masks = batch
            logits = model(masked_images)
            batch_size = masks.size(0)
            for i in range(batch_size):
                pred = logits[i]
                tgt = masks[i]
                iou = iou_score(pred, tgt)
                dice = dice_score(pred, tgt)
                total_iou += iou
                total_dice += dice
                total_items += 1

    if total_items == 0:
        print("No test items found.")
        return

    print(f"Eval over {total_items} samples — IoU={total_iou/total_items:.4f}, Dice={total_dice/total_items:.4f}")


if __name__ == "__main__":
    main()

import os
from pathlib import Path

import torch
import numpy as np
from PIL import Image

# ensure repository root is on sys.path so 'swu' can be imported
import sys
from pathlib import Path as _P
ROOT = str(_P(__file__).resolve().parents[1])
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from swu.dataset import WaterDataset
from swu.unet import UNet


def find_checkpoint():
    out = Path("outputs")
    candidates = [out / "unet_best.pth", out / "unet_epoch15.pth", out / "unet_epoch10.pth", out / "unet_epoch5.pth"]
    for c in candidates:
        if c.exists():
            return c
    return None


def save_side_by_side(mask, pred_prob, pred_bin, out_path: Path):
    # mask: (H,W) float 0..1
    # pred_prob: (H,W) float 0..1
    # pred_bin: (H,W) {0,1}
    H, W = mask.shape
    vis = np.zeros((H, W * 3, 3), dtype=np.uint8)

    def to_rgb(arr):
        a = (arr * 255.0).clip(0, 255).astype(np.uint8)
        return np.stack([a, a, a], axis=-1)

    vis[:, 0:W, :] = to_rgb(mask)
    vis[:, W : 2 * W, :] = to_rgb(pred_prob)
    vis[:, 2 * W : 3 * W, :] = to_rgb(pred_bin)

    Image.fromarray(vis).save(out_path)


def main():
    ds = WaterDataset(image_dir="data/images", mask_dir="data/masks")

    print(f"Dataset size: {len(ds)}")

    # check uniques for first 5 masks
    for i in range(min(5, len(ds))):
        _, _, mask = ds[i]
        uniq = torch.unique(mask)
        print(f"mask[{i}] unique: {uniq.tolist()} sum: {float(torch.sum(mask))}")

    ckpt = find_checkpoint()
    if ckpt is None:
        print("No checkpoint found in outputs/. Skipping model inference.")
        return

    print(f"Loading model from: {ckpt}")
    model = UNet(n_channels=3, n_classes=1)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    # Run inference on a single sample (index 0)
    img, masked_img, mask = ds[0]
    # tensors: image [3,H,W], mask [1,H,W]
    inp = masked_img.unsqueeze(0)
    with torch.no_grad():
        logits = model(inp)
        probs = torch.sigmoid(logits).squeeze().cpu()

    prob_min = float(probs.min())
    prob_max = float(probs.max())
    prob_mean = float(probs.mean())
    print(f"probs min={prob_min:.6f} max={prob_max:.6f} mean={prob_mean:.6f}")

    preds_bin = (probs >= 0.5).float()
    print(f"pred unique: {torch.unique(preds_bin).tolist()} sum: {float(torch.sum(preds_bin))}")

    tgt = mask.squeeze().cpu()
    print(f"target unique: {torch.unique(tgt).tolist()} sum: {float(torch.sum(tgt))}")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "pred_sample.png"
    save_side_by_side(tgt.numpy(), probs.numpy(), preds_bin.numpy(), out_path)
    print(f"Saved sample visualization to: {out_path}")


if __name__ == "__main__":
    main()

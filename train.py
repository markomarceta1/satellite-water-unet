import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

from swu.dataset import WaterDataset
from swu.metrics import dice_score, iou_score
from swu.unet import UNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a U-Net model on satellite water masks.")
    parser.add_argument("--image-dir", default="data/images", help="Path to directory with input images.")
    parser.add_argument("--mask-dir", default="data/masks", help="Path to directory with segmentation masks.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for training and validation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--val-split", type=float, default=0.15, help="Fraction of dataset reserved for validation.")
    parser.add_argument("--test-split", type=float, default=0.15, help="Fraction of dataset reserved for testing.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Compute device to use.")
    parser.add_argument("--output-dir", default="outputs", help="Directory where model checkpoints and logs are saved.")
    return parser.parse_args()


def get_dataloaders(args: argparse.Namespace):
    dataset = WaterDataset(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
    )
    print(f"Dataset size: {len(dataset)}")

    if args.val_split + args.test_split >= 1.0:
        raise ValueError("--val-split and --test-split must sum to less than 1.0")

    dataset_len = len(dataset)
    test_size = max(1, int(dataset_len * args.test_split))
    val_size = max(1, int(dataset_len * args.val_split))
    train_size = dataset_len - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    return train_loader, val_loader, test_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, criterion: nn.Module, device: torch.device):
    model.train()
    epoch_loss = 0.0

    for batch in loader:
        _, masked_images, masks = batch
        masked_images = masked_images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(masked_images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * masked_images.size(0)

    return epoch_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    epoch_loss = 0.0
    epoch_iou = 0.0
    epoch_dice = 0.0

    with torch.no_grad():
        for batch in loader:
            _, masked_images, masks = batch
            masked_images = masked_images.to(device)
            masks = masks.to(device)

            logits = model(masked_images)
            loss = criterion(logits, masks)

            epoch_loss += loss.item() * masked_images.size(0)
            epoch_iou += iou_score(logits, masks) * masked_images.size(0)
            epoch_dice += dice_score(logits, masks) * masked_images.size(0)

    size = len(loader.dataset)
    return epoch_loss / size, epoch_iou / size, epoch_dice / size


def save_checkpoint(model: nn.Module, output_dir: Path, epoch: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"unet_epoch{epoch}.pth"
    torch.save(model.state_dict(), path)
    return path


def save_best_checkpoint(model: nn.Module, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "unet_best.pth"
    torch.save(model.state_dict(), path)
    return path


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Loading dataset from: {args.image_dir} and {args.mask_dir}")

    train_loader, val_loader, test_loader = get_dataloaders(args)
    model = UNet(n_channels=3, n_classes=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)

    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}.")

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_path = None

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_iou, val_dice = evaluate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # save last epoch checkpoint
        _ = save_checkpoint(model, Path(args.output_dir), epoch)

        # save best model only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_best_checkpoint(model, Path(args.output_dir))

        print(
            f"Epoch {epoch}/{args.epochs}: "
            f"train_loss={train_loss:.4f}, "
            f"val_loss={val_loss:.4f}, "
            f"val_iou={val_iou:.4f}, "
            f"val_dice={val_dice:.4f}, "
            f"best_saved={(best_path.name if best_path is not None else 'none')}"
        )

    # plot losses
    try:
        plt.figure()
        plt.plot(train_losses, label="train_loss")
        plt.plot(val_losses, label="val_loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.tight_layout()
        plot_path = Path(args.output_dir) / "loss_curve.png"
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved loss curve to: {plot_path}")
    except Exception:
        print("Could not save loss plot (matplotlib may be unavailable).")

    # Final test evaluation using best model if available
    if best_path is not None:
        model.load_state_dict(torch.load(best_path, map_location=device))

    test_loss, test_iou, test_dice = evaluate(model, test_loader, criterion, device)
    print(
        f"Test results - loss={test_loss:.4f}, iou={test_iou:.4f}, dice={test_dice:.4f}"
    )

    print("Training complete.")


if __name__ == "__main__":
    main()

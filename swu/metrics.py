import torch


def _threshold_mask(predictions: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (predictions >= threshold).float()


def iou_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Compute the Intersection over Union score for binary segmentation."""
    predictions = torch.sigmoid(predictions)
    predictions = _threshold_mask(predictions, threshold)
    # Ensure targets are binary (0 or 1). Many masks are stored as 0-255
    # grayscale and converted to 0..1 by ToTensor(). Binarize here.
    targets = (targets >= 0.5).float()

    intersection = torch.sum(predictions * targets)
    union = torch.sum(predictions) + torch.sum(targets) - intersection
    return float((intersection + eps) / (union + eps))


def dice_score(predictions: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> float:
    """Compute the Dice coefficient for binary segmentation."""
    predictions = torch.sigmoid(predictions)
    predictions = _threshold_mask(predictions, threshold)
    # Binarize targets to 0/1 for correct overlap computation.
    targets = (targets >= 0.5).float()

    intersection = torch.sum(predictions * targets)
    return float((2 * intersection + eps) / (torch.sum(predictions) + torch.sum(targets) + eps))

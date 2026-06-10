"""Data loading and augmentation.

Expects an ImageFolder-style layout:

    data/
      train/<class_a>/*.jpg
      train/<class_b>/*.jpg
      val/<class_a>/*.jpg
      val/<class_b>/*.jpg
"""
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Standard ImageNet normalisation (matches the pretrained backbones).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_transforms(img_size: int, train: bool):
    """Return the transform pipeline for train (augmented) or eval (clean)."""
    if train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_dataloaders(data_dir, img_size: int, batch_size: int, num_workers: int):
    """Build train/val dataloaders. Returns (train_loader, val_loader, class_names)."""
    data_dir = Path(data_dir)
    train_dir, val_dir = data_dir / "train", data_dir / "val"
    for d in (train_dir, val_dir):
        if not d.exists():
            raise FileNotFoundError(
                f"Expected dataset folder '{d}'. Organise images as "
                "data/train/<class>/*.jpg and data/val/<class>/*.jpg."
            )

    train_ds = datasets.ImageFolder(train_dir, build_transforms(img_size, train=True))
    val_ds = datasets.ImageFolder(val_dir, build_transforms(img_size, train=False))

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, train_ds.classes

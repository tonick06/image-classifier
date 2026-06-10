"""Evaluate a trained checkpoint on the validation set.

    python -m src.evaluate --checkpoint models/best.pt
"""
import argparse

import torch
from sklearn.metrics import classification_report, confusion_matrix

from . import config
from .data import build_dataloaders
from .model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained model.")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--checkpoint", default=str(config.MODELS_DIR / "best.pt"))
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["class_names"]
    model = build_model(ckpt["model_name"], len(classes), freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    _, val_loader, _ = build_dataloaders(
        args.data_dir, ckpt["img_size"], args.batch_size, args.num_workers)

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            preds = model(images.to(device)).argmax(1).cpu()
            y_pred.extend(preds.tolist())
            y_true.extend(labels.tolist())

    print(classification_report(y_true, y_pred, target_names=classes))
    print("Confusion matrix (rows = true, cols = predicted):")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    main()

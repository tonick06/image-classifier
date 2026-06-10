"""Train an image classifier via transfer learning.

Run from the project root:
    python -m src.train --epochs 10 --model-name resnet18
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from . import config
from .data import build_dataloaders
from .model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Train an image classifier.")
    p.add_argument("--data-dir", default=str(config.DATA_DIR))
    p.add_argument("--model-name", default=config.MODEL_NAME)
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LR)
    p.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    p.add_argument("--num-workers", type=int, default=config.NUM_WORKERS)
    p.add_argument("--no-freeze", action="store_true",
                   help="Fine-tune the whole backbone instead of just the head.")
    p.add_argument("--out", default=str(config.MODELS_DIR / "best.pt"))
    return p.parse_args()


def run_epoch(model, loader, criterion, device, scaler, optimizer=None):
    """One pass over `loader`. Training if an optimizer is given, else eval."""
    is_train = optimizer is not None
    model.train(is_train)
    use_amp = device.type == "cuda"
    loss_sum, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, leave=False):
        images, labels = images.to(device), labels.to(device)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        loss_sum += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)

    return loss_sum / total, correct / total


def main():
    args = parse_args()
    torch.manual_seed(config.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_loader, val_loader, classes = build_dataloaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers)
    print(f"Classes ({len(classes)}): {classes}")

    model = build_model(args.model_name, len(classes),
                        freeze_backbone=not args.no_freeze).to(device)
    criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, scaler, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device, scaler)
        print(f"Epoch {epoch:02d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} "
              f"| val loss {va_loss:.3f} acc {va_acc:.3f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({
                "model_state": model.state_dict(),
                "class_names": classes,
                "model_name": args.model_name,
                "img_size": args.img_size,
            }, out_path)
            print(f"  -> saved new best ({best_acc:.3f}) to {out_path}")

    print(f"Done. Best val accuracy: {best_acc:.3f}")


if __name__ == "__main__":
    main()

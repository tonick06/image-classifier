"""Predict the class of a single image.

    python -m src.predict path/to/image.jpg --topk 3
"""
import argparse

import torch
import torch.nn.functional as F
from PIL import Image

from . import config
from .data import build_transforms
from .model import build_model


def parse_args():
    p = argparse.ArgumentParser(description="Predict on a single image.")
    p.add_argument("image", help="Path to an image file.")
    p.add_argument("--checkpoint", default=str(config.MODELS_DIR / "best.pt"))
    p.add_argument("--topk", type=int, default=3)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location=device)
    classes = ckpt["class_names"]
    model = build_model(ckpt["model_name"], len(classes), freeze_backbone=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    transform = build_transforms(ckpt["img_size"], train=False)
    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        probs = F.softmax(model(x), dim=1)[0].cpu()

    k = min(args.topk, len(classes))
    confidences, indices = probs.topk(k)
    print(f"Predictions for {args.image}:")
    for conf, idx in zip(confidences.tolist(), indices.tolist()):
        print(f"  {classes[idx]:20s} {conf * 100:5.1f}%")


if __name__ == "__main__":
    main()

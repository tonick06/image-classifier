# Image Classifier (Transfer Learning)

A PyTorch image classifier built with transfer learning: a pretrained CNN
backbone with a fresh classification head trained on your own dataset. Runs
locally on an RTX 3070 with mixed precision enabled automatically.

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. Install PyTorch with CUDA (pick the command for your CUDA version from
#    https://pytorch.org/get-started/locally/ ). For recent NVIDIA drivers:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install the rest
pip install -r requirements.txt

# 4. Verify the GPU is visible
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Dataset

Organise images into class subfolders, split into `train` and `val`:

```
data/
  train/cardboard/*.jpg
  train/glass/*.jpg
  val/cardboard/*.jpg
  val/glass/*.jpg
```

Some good domains to grab from Kaggle:
- **Recycling / waste sorting** (e.g. TrashNet)
- **Plant disease detection** (e.g. PlantVillage)
- **Food classification** (e.g. Food-101)

If a dataset isn't pre-split, put everything under class folders and write a
quick script to move ~20% of each class into `val/` (a good first task for
Claude Code — see `CLAUDE.md`).

## Usage

```bash
# Train (head-only, fast)
python -m src.train --epochs 10 --model-name resnet18

# Fine-tune the whole network (slower, usually higher accuracy)
python -m src.train --epochs 15 --no-freeze --lr 1e-4

# Evaluate the best checkpoint
python -m src.evaluate --checkpoint models/best.pt

# Predict on one image
python -m src.predict path/to/image.jpg --topk 3
```

## Notes
- Mixed precision (AMP) turns on automatically when CUDA is available.
- Out-of-memory? Lower `--batch-size` or `--img-size`.
- Models: `resnet18` (fast), `resnet50`, `mobilenet_v3_large`, `efficientnet_b0`.

See `CLAUDE.md` for the build roadmap.

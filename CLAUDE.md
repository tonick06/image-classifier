# CLAUDE.md

Context for Claude Code working in this repo.

## What this is
An image classifier built with **transfer learning** in PyTorch. A pretrained
CNN backbone (ResNet/MobileNet/EfficientNet) is frozen, its final layer is
replaced with a fresh head, and only the head is trained on a custom dataset.
Goal: a clean, CV-worthy ML project that trains locally on an RTX 3070 and
ships with a working predict/eval path and (later) a small demo UI.

## Stack
- Python 3.10+, PyTorch (CUDA), torchvision
- scikit-learn (metrics), matplotlib, tqdm, Pillow

## Layout
```
src/
  config.py    # default hyperparameters & paths
  data.py      # ImageFolder dataloaders + transforms
  model.py     # build_model(): pretrained backbone + new head
  train.py     # training loop (AMP, validation, checkpointing)
  evaluate.py  # classification report + confusion matrix
  predict.py   # single-image inference
data/          # train/<class>/*.jpg and val/<class>/*.jpg  (gitignored)
models/        # saved checkpoints, e.g. best.pt            (gitignored)
```

## Conventions
- `src/` is a package; run scripts as modules from the project root:
  `python -m src.train`, `python -m src.evaluate`, `python -m src.predict`.
- Scripts are argparse-driven; defaults live in `src/config.py`.
- Checkpoints store `model_state`, `class_names`, `model_name`, `img_size`
  so eval/predict can rebuild the model without extra flags.
- AMP (mixed precision) auto-enables on CUDA, off on CPU — no flag needed.

## How to run
```bash
python -m src.train --epochs 10 --model-name resnet18
python -m src.evaluate --checkpoint models/best.pt
python -m src.predict path/to/image.jpg --topk 3
```

## Roadmap / TODO (in order)
1. **Get data in.** Pick a domain and populate `data/train` and `data/val`
   (see README for dataset options + the split helper idea).
2. **Baseline.** Train resnet18 head-only for ~10 epochs; record val accuracy.
3. **Improve.** Try `--no-freeze` fine-tuning with a lower LR; try
   mobilenet_v3_large / efficientnet_b0; add a LR scheduler and early stopping.
4. **Report.** Save the confusion matrix as a figure; log metrics per run.
5. **Demo (CV polish).** Wrap `predict` in a small FastAPI endpoint or a
   Streamlit upload-an-image UI, then deploy and link it in the README.

## Notes / gotchas
- Don't commit datasets or `.pt` files — they're gitignored on purpose.
- If you hit CUDA out-of-memory, lower `--batch-size` or `--img-size`.
- Keep the checkpoint schema stable, or update eval/predict together.

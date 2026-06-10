"""Central configuration defaults.

CLI flags in the scripts override anything here, so this is just the
single place to change sensible defaults.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

# Model: one of resnet18 | resnet50 | mobilenet_v3_large | efficientnet_b0
MODEL_NAME = "resnet18"
FREEZE_BACKBONE = True  # train only the new head first (fast, light on VRAM)

# Data / training
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
EPOCHS = 10
LR = 1e-3
WEIGHT_DECAY = 1e-4
SEED = 42

"""Download Food-101 and arrange it into data/train/<class> and data/val/<class>.

    python -m scripts.prepare_food101
"""
from torchvision.datasets import Food101

from src import config

RAW_DIR = config.DATA_DIR / "_food101_raw"


def link_split(dataset: Food101, dest_root) -> None:
    for image_file, label in zip(dataset._image_files, dataset._labels):
        class_dir = dest_root / dataset.classes[label]
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / image_file.name
        if not dest.exists():
            dest.hardlink_to(image_file)


def main():
    print("Downloading/extracting Food-101 (~5 GB, this may take a while)...")
    train_ds = Food101(root=RAW_DIR, split="train", download=True)
    test_ds = Food101(root=RAW_DIR, split="test", download=True)
    print(f"Classes: {len(train_ds.classes)} | train: {len(train_ds)} | val: {len(test_ds)}")

    print("Linking train split into data/train ...")
    link_split(train_ds, config.DATA_DIR / "train")
    print("Linking val split into data/val ...")
    link_split(test_ds, config.DATA_DIR / "val")

    print("Done. data/train and data/val are ready.")
    print(f"You can delete {RAW_DIR} to free disk space; the hardlinked files in "
          "data/train and data/val will remain intact.")


if __name__ == "__main__":
    main()

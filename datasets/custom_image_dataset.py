"""
Custom PyTorch Image Dataset

Reads image file paths and labels from a CSV file (columns: filepath,label) and
returns (image, label) pairs. The image is loaded with PIL and an optional
transform can be applied. Target transform is also supported.

Example CSV format (header optional):
filepath,label
/path/to/img1.png,0
/path/to/img2.png,1

This module intentionally avoids pandas dependency and uses the csv module.
"""
from __future__ import annotations

import csv
import os
from typing import Callable, List, Optional, Sequence, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset


def default_image_loader(path: str) -> Image.Image:
    """Open an image file and convert to RGB."""
    with Image.open(path) as img:
        return img.convert("RGB")


class CustomImageDataset(Dataset):
    """A simple custom Dataset for image classification.

    Args:
        csv_file: Path to a CSV file containing rows of `filepath,label`.
                  The CSV may optionally have a header. Filepaths may be
                  absolute or relative; relative paths are resolved relative
                  to the CSV file's directory.
        transform: Optional callable applied to the PIL image (e.g., to convert
                   to tensor / do augmentations).
        target_transform: Optional callable applied to the target/label.
        loader: Optional callable that takes a path and returns a PIL Image.
                Defaults to a loader that opens the image and converts to RGB.
    """

    def __init__(
        self,
        csv_file: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        loader: Optional[Callable[[str], Image.Image]] = None,
    ) -> None:
        if not os.path.isfile(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")

        self.csv_file = csv_file
        self.root = os.path.dirname(os.path.abspath(csv_file))
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader or default_image_loader

        self.samples: List[Tuple[str, int]] = []
        self._load_csv()

        if len(self.samples) == 0:
            raise ValueError(f"No samples found in CSV: {csv_file}")

    def _load_csv(self) -> None:
        with open(self.csv_file, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)

        # If the first row looks like a header, skip it. Header detection is
        # simple: if any non-numeric characters in the label column of the
        # first row, assume header.
        if len(rows) == 0:
            return

        # Normalize rows: if first row has 'filepath' or 'label' keywords, drop it
        first = rows[0]
        if len(first) >= 2 and (
            "filepath" in first[0].lower() or "label" in first[1].lower()
        ):
            rows = rows[1:]

        for row in rows:
            if len(row) < 2:
                continue
            path, label = row[0].strip(), row[1].strip()
            if path == "":
                continue

            # Resolve relative paths relative to CSV file
            if not os.path.isabs(path):
                path = os.path.join(self.root, path)

            try:
                label_int = int(label)
            except Exception:
                # Try float -> int if possible
                try:
                    label_int = int(float(label))
                except Exception:
                    raise ValueError(f"Label is not numeric: {label}")

            self.samples.append((path, label_int))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self.samples):
            raise IndexError("Index out of range")

        path, label = self.samples[idx]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Image file not found: {path}")

        img = self.loader(path)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)

        return img, label


__all__ = ["CustomImageDataset"]

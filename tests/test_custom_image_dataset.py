import csv
import os
import tempfile
from typing import Tuple

import numpy as np
from PIL import Image
import torch

from datasets.custom_image_dataset import CustomImageDataset


def make_sample_image(path: str, color: Tuple[int, int, int] = (255, 0, 0)) -> None:
    arr = np.zeros((8, 8, 3), dtype=np.uint8)
    arr[:, :] = color
    img = Image.fromarray(arr)
    img.save(path)


def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.array(img, dtype=np.float32) / 255.0  # H,W,C
    # Convert to C,H,W
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def test_dataset_basic(tmp_path):
    # Create 3 small images
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    paths = []
    labels = [0, 1, 2]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

    for i, color in enumerate(colors):
        p = img_dir / f"img_{i}.png"
        make_sample_image(str(p), color=color)
        paths.append(str(p))

    # Create CSV
    csv_path = tmp_path / "samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])  # header
        for p, l in zip(paths, labels):
            writer.writerow([p, l])

    # Instantiate dataset WITHOUT transform
    ds = CustomImageDataset(str(csv_path))
    assert len(ds) == 3

    img0, label0 = ds[0]
    assert isinstance(img0, Image.Image)
    assert label0 == 0

    # Instantiate dataset WITH transform that converts to torch tensor
    ds_t = CustomImageDataset(str(csv_path), transform=pil_to_tensor)
    img1, label1 = ds_t[1]
    assert isinstance(img1, torch.Tensor)
    assert img1.shape[0] == 3  # C dimension
    assert label1 == 1


def test_target_transform(tmp_path):
    img_dir = tmp_path / "images2"
    img_dir.mkdir()

    p = img_dir / "img.png"
    make_sample_image(str(p), color=(10, 20, 30))

    csv_path = tmp_path / "samples2.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "label"])  # header
        writer.writerow([str(p), 3])

    ds = CustomImageDataset(str(csv_path), transform=pil_to_tensor, target_transform=lambda x: x * 10)
    img, label = ds[0]
    assert isinstance(img, torch.Tensor)
    assert label == 30

"""
Which vision dataset federated training uses (shared by FastAPI + Flower subprocess).

Set env **FL_DATASET** (or `python main.py --dataset ...`):

- **emnist** (default) — EMNIST ByClass via tfds, 62 classes.
- **fashion_mnist** — Fashion-MNIST via Keras, 10 classes.
- **kmnist** — KMNIST via tfds, 10 classes.

All use 28×28×1 so the same CNN tiers apply.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    num_classes: int
    input_shape: Tuple[int, int, int]
    mean: float
    std: float
    transpose_hw: bool
    description: str


def _env_dataset_key() -> str:
    return os.environ.get("FL_DATASET", "emnist").strip().lower().replace("-", "_")


def get_dataset_spec() -> DatasetSpec:
    k = _env_dataset_key()
    if k in ("emnist", "emnist_byclass"):
        return DatasetSpec(
            key="emnist",
            num_classes=62,
            input_shape=(28, 28, 1),
            mean=0.1307,
            std=0.3081,
            transpose_hw=True,
            description="EMNIST ByClass (62 classes)",
        )
    if k in ("fashion_mnist", "fashion"):
        return DatasetSpec(
            key="fashion_mnist",
            num_classes=10,
            input_shape=(28, 28, 1),
            mean=0.2860406,
            std=0.35302424,
            transpose_hw=False,
            description="Fashion-MNIST (10 classes)",
        )
    if k in ("kmnist",):
        return DatasetSpec(
            key="kmnist",
            num_classes=10,
            input_shape=(28, 28, 1),
            mean=0.1904,
            std=0.3475,
            transpose_hw=False,
            description="KMNIST (10 classes)",
        )
    print(f"[fl.datasets_config] Unknown FL_DATASET={k!r}, using emnist")
    return DatasetSpec(
        key="emnist",
        num_classes=62,
        input_shape=(28, 28, 1),
        mean=0.1307,
        std=0.3081,
        transpose_hw=True,
        description="EMNIST ByClass (62 classes)",
    )


FASHION_LABELS = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def label_to_str(idx: int) -> str:
    spec = get_dataset_spec()
    if spec.key == "fashion_mnist":
        if 0 <= idx < len(FASHION_LABELS):
            return FASHION_LABELS[idx]
        return f"class_{idx}"
    if spec.key == "kmnist":
        return f"kmnist_{idx}"
    if idx < 10:
        return chr(ord("0") + idx)
    if idx < 36:
        return chr(ord("a") + (idx - 10))
    return chr(ord("A") + (idx - 36))

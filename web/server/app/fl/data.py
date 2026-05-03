"""
Federated data: configurable 28×28 grayscale datasets + non-IID partitioning + tf.data.

Env **FL_DATASET**: `emnist` | `fashion_mnist` | `kmnist` (see `datasets_config.py`).
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import tensorflow as tf

from .datasets_config import get_dataset_spec

_DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")


def num_classes() -> int:
    return get_dataset_spec().num_classes


def input_shape() -> Tuple[int, int, int]:
    return get_dataset_spec().input_shape


def needs_sketch_spatial_align() -> bool:
    """True only for EMNIST (tfds column-major); sketches must match training transpose."""
    return get_dataset_spec().transpose_hw


def align_sketch_like_training(image_hw: np.ndarray) -> np.ndarray:
    if image_hw.ndim != 2:
        raise ValueError("align_sketch_like_training expects a 2D array")
    if not needs_sketch_spatial_align():
        return image_hw
    return np.transpose(image_hw, (1, 0))


def _normalize(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    spec = get_dataset_spec()
    image = tf.cast(image, tf.float32) / 255.0
    image = (image - spec.mean) / spec.std
    if spec.transpose_hw:
        image = tf.transpose(image, perm=[1, 0, 2])
    return image, tf.cast(label, tf.int32)


@lru_cache(maxsize=8)
def _load_arrays_cached(dataset_key: str, cache_dir: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    os.makedirs(cache_dir, exist_ok=True)
    spec = get_dataset_spec()
    if spec.key != dataset_key:
        print(f"[fl.data] Warning: cache key {dataset_key!r} != active spec {spec.key!r}")

    if spec.key == "emnist":
        try:
            import tensorflow_datasets as tfds  # type: ignore

            ds_train, ds_test = tfds.load(
                "emnist/byclass",
                split=["train", "test"],
                as_supervised=True,
                data_dir=cache_dir,
            )
            x_train, y_train = _tfds_to_numpy(ds_train)
            x_test, y_test = _tfds_to_numpy(ds_test)
            print(f"[fl.data] EMNIST byclass train={len(x_train)} test={len(x_test)}")
            return x_train, y_train, x_test, y_test
        except Exception as exc:
            print(f"[fl.data] EMNIST failed ({exc}); MNIST fallback")
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            x_train = x_train[..., np.newaxis].astype(np.uint8)
            x_test = x_test[..., np.newaxis].astype(np.uint8)
            return x_train, y_train.astype(np.int32), x_test, y_test.astype(np.int32)

    if spec.key == "fashion_mnist":
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
        x_train = x_train[..., np.newaxis].astype(np.uint8)
        x_test = x_test[..., np.newaxis].astype(np.uint8)
        print(f"[fl.data] Fashion-MNIST train={len(x_train)} test={len(x_test)}")
        return x_train, y_train.astype(np.int32), x_test, y_test.astype(np.int32)

    if spec.key == "kmnist":
        import tensorflow_datasets as tfds  # type: ignore

        ds_train, ds_test = tfds.load(
            "kmnist",
            split=["train", "test"],
            as_supervised=True,
            data_dir=cache_dir,
        )
        x_train, y_train = _tfds_to_numpy(ds_train)
        x_test, y_test = _tfds_to_numpy(ds_test)
        print(f"[fl.data] KMNIST train={len(x_train)} test={len(x_test)}")
        return x_train, y_train, x_test, y_test

    raise ValueError(f"Unknown dataset {spec.key}")


def _tfds_to_numpy(ds: "tf.data.Dataset") -> Tuple[np.ndarray, np.ndarray]:
    images, labels = [], []
    for img, lbl in ds.as_numpy_iterator():
        images.append(img)
        labels.append(lbl)
    return np.stack(images), np.array(labels, dtype=np.int32)


def _load_arrays(cache_dir: str = _DEFAULT_CACHE_DIR) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _load_arrays_cached(get_dataset_spec().key, cache_dir)


def _stable_seed(client_id: str) -> int:
    return abs(hash(client_id)) % (2**32)


def _dirichlet_partition(
    labels: np.ndarray, num_clients: int, alpha: float, seed: int, n_cls: int
) -> Dict[int, np.ndarray]:
    rng = np.random.default_rng(seed)
    client_indices: Dict[int, list] = {i: [] for i in range(num_clients)}

    for cls in range(n_cls):
        cls_idx = np.where(labels == cls)[0]
        if len(cls_idx) == 0:
            continue
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet([alpha] * num_clients)
        splits = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        for client_id, chunk in enumerate(np.split(cls_idx, splits)):
            client_indices[client_id].extend(chunk.tolist())

    return {cid: np.array(idx, dtype=np.int64) for cid, idx in client_indices.items()}


def load_partition(
    client_id: str,
    num_clients: int = 4,
    alpha: float = 0.5,
    val_fraction: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_cls = num_classes()
    x_train_full, y_train_full, _, _ = _load_arrays()

    partition_seed = 1234 + int(num_clients) + int(alpha * 100)
    parts = _dirichlet_partition(y_train_full, num_clients, alpha, partition_seed, n_cls)

    slot = _stable_seed(client_id) % num_clients
    idx = parts.get(slot, np.array([], dtype=np.int64))

    if len(idx) == 0:
        idx = np.arange(min(256, len(x_train_full)))

    x = x_train_full[idx]
    y = y_train_full[idx]

    rng = np.random.default_rng(_stable_seed(client_id))
    perm = rng.permutation(len(x))
    n_val = max(1, int(len(x) * val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
) -> tf.data.Dataset:
    autotune = tf.data.AUTOTUNE
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(x), 2048), reshuffle_each_iteration=True)
    ds = ds.map(_normalize, num_parallel_calls=autotune)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(autotune)
    return ds


def evaluate_dataset(
    x: np.ndarray, y: np.ndarray, batch_size: int = 256
) -> tf.data.Dataset:
    return make_dataset(x, y, batch_size=batch_size, shuffle=False)


def load_global_eval_set(max_samples: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
    _, _, x_test, y_test = _load_arrays()
    if len(x_test) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(x_test), size=max_samples, replace=False)
        x_test, y_test = x_test[idx], y_test[idx]
    return x_test, y_test

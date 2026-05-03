"""
Tiered Keras CNNs for 28×28×1 federated classification (dataset via FL_DATASET).
"""

from __future__ import annotations

from typing import Dict, List, Literal

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .datasets_config import get_dataset_spec, label_to_str

# Back-compat for Gradio / imports
idx_to_char = label_to_str

TIER_CONFIGS: Dict[str, Dict] = {
    "lite": {
        "conv_filters": [16, 32],
        "dense_units": 128,
        "dropout": 0.2,
        "use_batchnorm": False,
        "use_gap": False,
        "description": "Lightweight CNN for low-RAM/battery devices",
    },
    "standard": {
        "conv_filters": [32, 64, 128],
        "dense_units": 256,
        "dropout": 0.3,
        "use_batchnorm": True,
        "use_gap": False,
        "description": "Balanced CNN for normal operation",
    },
    "full": {
        "conv_filters": [32, 64, 128, 256],
        "dense_units": 256,
        "dropout": 0.4,
        "use_batchnorm": True,
        "use_gap": True,
        "description": "Deep CNN for high-end / GPU / charging devices",
    },
}


def _build_keras_model(tier: str) -> keras.Model:
    cfg = TIER_CONFIGS[tier]
    spec = get_dataset_spec()
    n_cls = spec.num_classes
    in_shape = spec.input_shape

    model = keras.Sequential(name=f"fl_{spec.key}_{tier}")
    model.add(keras.Input(shape=in_shape))

    for i, filters in enumerate(cfg["conv_filters"]):
        model.add(layers.Conv2D(filters, 3, padding="same", use_bias=not cfg["use_batchnorm"]))
        if cfg["use_batchnorm"]:
            model.add(layers.BatchNormalization())
        model.add(layers.ReLU())
        if (i + 1) % 1 == 0 and (i < len(cfg["conv_filters"]) - 1):
            model.add(layers.MaxPooling2D(2))

    if cfg["use_gap"]:
        model.add(layers.GlobalAveragePooling2D())
    else:
        model.add(layers.MaxPooling2D(2))
        model.add(layers.Flatten())

    model.add(layers.Dense(cfg["dense_units"], activation="relu"))
    model.add(layers.Dropout(cfg["dropout"]))
    model.add(layers.Dense(n_cls, activation="softmax"))

    return model


def build_model(
    tier: Literal["lite", "standard", "full"] = "standard",
    distribute: bool = False,
    learning_rate: float = 1e-3,
) -> keras.Model:
    if tier not in TIER_CONFIGS:
        raise ValueError(f"Unknown tier '{tier}'. Choose from {list(TIER_CONFIGS)}")

    if distribute and len(tf.config.list_physical_devices("GPU")) > 1:
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            model = _build_keras_model(tier)
            model.compile(
                optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
    else:
        model = _build_keras_model(tier)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

    return model


def model_size_kb(model: keras.Model) -> float:
    return float(model.count_params()) * 4 / 1024


def get_initial_weights(tier: str) -> List[np.ndarray]:
    return build_model(tier).get_weights()

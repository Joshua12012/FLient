"""
Persist global model snapshots after each FL round.

Two artifact types per tier (adaptive serving):
- `<tier>.weights.h5` : full-precision Keras weights (best accuracy)
- `<tier>.tflite`     : float16-quantized TFLite (small + fast for low-tier devices)

The Gradio client then chooses which to download based on its tier.
"""

from __future__ import annotations

import os
import tempfile
from typing import List, Optional

import numpy as np
import tensorflow as tf

from .model import build_model

_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models"
)


def models_dir() -> str:
    os.makedirs(_MODELS_DIR, exist_ok=True)
    return _MODELS_DIR


def weights_path(tier: str) -> str:
    return os.path.join(models_dir(), f"global_{tier}.weights.h5")


def tflite_path(tier: str) -> str:
    return os.path.join(models_dir(), f"global_{tier}.tflite")


def save_keras_weights(tier: str, weights: List[np.ndarray]) -> str:
    """Build a fresh model, set weights, and save .weights.h5."""
    model = build_model(tier)
    model.set_weights(weights)
    out = weights_path(tier)
    model.save_weights(out)
    return out


def export_tflite_float16(tier: str, weights: List[np.ndarray]) -> Optional[str]:
    """
    Convert the weighted model to float16 TFLite (good size/accuracy tradeoff).
    Returns the file path on success, or None if conversion failed.
    """
    try:
        model = build_model(tier)
        model.set_weights(weights)

        # TFLite converter requires a SavedModel or Keras model.
        with tempfile.TemporaryDirectory() as tmp:
            saved = os.path.join(tmp, "saved")
            model.export(saved)  # SavedModel export
            converter = tf.lite.TFLiteConverter.from_saved_model(saved)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.float16]
            tflite_blob = converter.convert()

        out = tflite_path(tier)
        with open(out, "wb") as f:
            f.write(tflite_blob)
        return out
    except Exception as exc:
        print(f"[fl.inference] TFLite export failed for tier '{tier}': {exc}")
        return None


def save_round_artifacts(tier: str, weights: List[np.ndarray]) -> dict:
    """
    Save both Keras weights + (best-effort) TFLite float16 for this tier.
    Returns a dict describing the saved file paths and sizes.
    """
    keras_p = save_keras_weights(tier, weights)
    tflite_p = export_tflite_float16(tier, weights)

    info = {
        "tier": tier,
        "keras_weights": keras_p,
        "keras_size_bytes": os.path.getsize(keras_p) if os.path.exists(keras_p) else 0,
        "tflite_path": tflite_p,
        "tflite_size_bytes": os.path.getsize(tflite_p) if tflite_p and os.path.exists(tflite_p) else 0,
    }
    return info


def load_keras_weights(tier: str) -> Optional[List[np.ndarray]]:
    """Read back the latest weights for a tier (used by FL server warm-start)."""
    p = weights_path(tier)
    if not os.path.exists(p):
        return None
    model = build_model(tier)
    model.load_weights(p)
    return model.get_weights()

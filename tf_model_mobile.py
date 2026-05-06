"""
tf_model_mobile.py  —  TensorFlow/TFLite models for mobile federated learning
Converted from PyTorch models for on-device training with TFLite

NUM_CLASSES = 62 (FEMNIST: digits + upper + lower)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NUM_CLASSES = 62
FEATURE_DIM = 128


# ══════════════════════════════════════════════════════════════════════════════
# SMALL MODEL — For mobile devices (lightweight, fast training)
# ══════════════════════════════════════════════════════════════════════════════

def build_small_model(input_shape=(28, 28, 1)):
    """
    Minimal model: single conv block + one FC.
    ~50K parameters - suitable for mobile training.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    x = layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(4)(x)  # 28x28 -> 7x7
    
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="EdgeCNN_Small")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# MEDIUM MODEL — For better phones
# ══════════════════════════════════════════════════════════════════════════════

def build_medium_model(input_shape=(28, 28, 1)):
    """
    Lighter model: fewer filters, no BatchNorm, single FC layer.
    ~150K parameters - good for mid-range phones.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D(2)(x)  # 28x28 -> 14x14
    
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(2)(x)  # 14x14 -> 7x7
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="EdgeCNN_Medium")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# LARGE MODEL — For high-end phones
# ══════════════════════════════════════════════════════════════════════════════

def build_large_model(input_shape=(28, 28, 1)):
    """
    Full CNN with BatchNorm.
    ~400K parameters - for high-end phones only.
    """
    inputs = keras.Input(shape=input_shape, name="input")
    
    # Block 1
    x = layers.Conv2D(32, 3, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(32, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.1)(x)
    
    # Block 2
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.1)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(FEATURE_DIM, activation="relu", name="features")(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(NUM_CLASSES, name="logits")(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name="EdgeCNN_Large")
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Factory helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_model(variant: str = "small", input_shape=(28, 28, 1)):
    """
    Returns the model for a given variant.
    variant: 'small' | 'medium' | 'large'
    """
    models = {
        "small": build_small_model,
        "medium": build_medium_model,
        "large": build_large_model,
    }
    if variant not in models:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(models)}")
    return models[variant](input_shape)


def convert_to_tflite(model, output_path="model.tflite", quantize=True):
    """
    Convert TensorFlow model to TFLite format for mobile deployment.
    
    Args:
        model: TensorFlow Keras model
        output_path: Path to save TFLite model
        quantize: Whether to apply quantization (reduces size by ~4x)
    """
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if quantize:
        # Apply post-training quantization
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]  # Or tf.int8 for more compression
    
    tflite_model = converter.convert()
    
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    
    print(f"TFLite model saved to {output_path}")
    print(f"Model size: {len(tflite_model) / 1024:.2f} KB")
    
    return output_path


def count_parameters(model):
    """Count trainable parameters in the model."""
    return sum([tf.size(w).numpy() for w in model.trainable_weights])


# ══════════════════════════════════════════════════════════════════════════════
# Test the models
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import numpy as np
    
    # Test all model variants
    x = np.random.randn(4, 28, 28, 1).astype(np.float32)
    
    print("=" * 60)
    print("TensorFlow Model Test")
    print("=" * 60)
    
    for variant in ["small", "medium", "large"]:
        model = get_model(variant)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        
        out = model(x, training=False)
        params = count_parameters(model)
        
        print(f"\n{variant.upper()} Model:")
        print(f"  Parameters: {params:,}")
        print(f"  Output shape: {out.shape}")
        
        # Test TFLite conversion
        tflite_path = f"model_{variant}.tflite"
        convert_to_tflite(model, tflite_path, quantize=False)
        convert_to_tflite(model, f"model_{variant}_quant.tflite", quantize=True)
    
    print("\n" + "=" * 60)
    print("All models tested successfully!")
    print("=" * 60)

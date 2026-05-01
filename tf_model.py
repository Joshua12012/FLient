"""
TensorFlow / TFLite helpers for mobile runtime support.

This module is optional and provides a TensorFlow equivalent of the
PyTorch models, plus a helper to export a Keras model to TFLite.

Use this if you want to produce a `.tflite` client model for mobile deployment.
"""

import tensorflow as tf

NUM_CLASSES = 62
FEATURE_DIM = 128


def build_large_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="input")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(FEATURE_DIM, activation="relu", name="features")(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_medium_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="input")
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D(2)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def build_small_model():
    inputs = tf.keras.Input(shape=(28, 28, 1), name="input")
    x = tf.keras.layers.Conv2D(8, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPool2D(4)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, name="logits")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def get_tf_model(variant="large"):
    variants = {
        "large": build_large_model,
        "medium": build_medium_model,
        "small": build_small_model,
    }
    if variant not in variants:
        raise ValueError(f"Unknown variant '{variant}'")
    return variants[variant]()


def export_tflite(model, output_path="model.tflite", quantize=False):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, "wb") as f:
        f.write(tflite_model)
    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Export a TensorFlow model to TFLite")
    parser.add_argument("--variant", type=str, default="small",
                        choices=["large", "medium", "small"])
    parser.add_argument("--output", type=str, default="model.tflite")
    parser.add_argument("--quantize", action="store_true")
    args = parser.parse_args()

    model = get_tf_model(args.variant)
    print(f"Building TensorFlow model: {args.variant}")
    output_path = export_tflite(model, args.output, quantize=args.quantize)
    print(f"Exported TFLite model to {output_path}")

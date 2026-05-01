"""
TFLite client runtime skeleton.

This is an example mobile client process for loading a `.tflite` model
and running inference on device. It is not a full Flower training client,
but it shows the correct mobile runtime for a lightweight deployment.
"""

print("TFLite client module loaded")

import argparse
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
    print("Using tflite-runtime (mobile)")
except ImportError:
    try:
        from tensorflow.lite import Interpreter
        print("Using tensorflow.lite (desktop fallback)")
    except ImportError:
        raise ImportError("Neither tflite-runtime nor tensorflow.lite available. Install one of them.")


def load_interpreter(model_path: str):
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def run_inference(interpreter, image: np.ndarray):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_data = image.astype(np.float32)
    input_data = np.expand_dims(input_data, axis=0)

    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]["index"])
    return output_data


def main():
    print("Starting TFLite client...")
    parser = argparse.ArgumentParser(description="Run a simple TFLite mobile client demo")
    parser.add_argument("--model", type=str, default="model.tflite",
                        help="Path to the TFLite model file")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    interpreter = load_interpreter(args.model)
    print(f"Loaded TFLite model: {args.model}")
    print("TFLite client ready for inference")
    return interpreter


print("About to check __name__")
if __name__ == "__main__":
    print("Entering main block")
    interpreter = main()

    dummy_image = np.random.rand(28, 28, 1).astype(np.float32)
    output = run_inference(interpreter, dummy_image)
    print(f"Inference output shape: {output.shape}")
    print(f"Output sample: {output[0][:5]}")

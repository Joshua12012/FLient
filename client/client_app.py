# import os
# import requests
# import torch
# from pathlib import Path
# from executorch.runtime import Runtime, Program, Method
# from dotenv import load_dotenv
# load_dotenv()
# # 1. Configuration
# HOST = os.getenv("TAILSCALE_IP")
# PORT = os.getenv("PORT")

# # SERVER_URL = f"http://{HOST}:{PORT}/download_model"
# SERVER_URL = f"http://127.0.0.1:8001/download_model"
# LOCAL_MODEL_PATH = Path("downloaded_global_model.pte")

# def download_global_model():
#     """Fetches the latest compiled ExecuTorch model from the central coordinator server."""
#     print(f"Connecting to central server at: {SERVER_URL}...")
#     try:
#         response = requests.get(SERVER_URL, stream=True)
#         if response.status_code == 200:
#             with open(LOCAL_MODEL_PATH, 'wb') as f:
#                 for chunk in response.iter_content(chunk_size=8192):
#                     f.write(chunk)
#             print(f"Success! Global model downloaded and cached at: {LOCAL_MODEL_PATH}")
#             return True
#         else:
#             print(f"Server returned error code: {response.status_code}")
#             return False
#     except requests.exceptions.ConnectionError:
#         print("Failed to connect to the server. Is server_app.py running?")
#         return False


# def run_local_inference():
#     """Loads the downloaded binary and runs an optimized native forward pass."""
#     if not LOCAL_MODEL_PATH.exists():
#         print("Cannot run inference; local model binary is missing.")
#         return

#     print("Initializing ExecuTorch Runtime Engine...")
#     et_runtime = Runtime.get()
    
#     print("Loading network graph into memory layout...")
#     program = et_runtime.load_program(LOCAL_MODEL_PATH)
    
#     # Load the execution method
#     forward_method = program.load_method("forward")
    
#     # Generate mock local sensor/feature data matching the 10-feature structure
#     # In production, this would be real data collected on-device
#     local_sensor_input = (torch.ones(1, 10),)
    
#     print("🚀 Running low-overhead C++ forward pass...")
#     outputs = forward_method.execute(local_sensor_input)
    
#     print("\n Edge Inference Successful!")
#     print(f"Resulting Predictions Layout (4 Classes): \n{outputs[0]}\n")

# if __name__ == "__main__":
#     # Run the over-the-air deployment loop
#     if download_global_model():
#         run_local_inference()

import os
import io
import sys
import requests
import torch
from pathlib import Path
from executorch.runtime import Runtime

# 1. Dynamically append the project root directory to the Python Path
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent  # Points directly to /home/joshua/FLient
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))


from federated_server.export_global_model import FederatedGlobalModel

# SERVER_URL = f"http://{HOST}:{PORT}/download_model"
SERVER_URL = f"http://127.0.0.1:8001/download_model"
# Configuration
SERVER_BASE_URL = "http://127.0.0.1:8001"
LOCAL_MODEL_PATH = Path("downloaded_global_model.pte")

def download_global_model():
    print(f"Downloading global model from {SERVER_BASE_URL}/download_model...")
    response = requests.get(f"{SERVER_BASE_URL}/download_model", stream=True)
    if response.status_code == 200:
        with open(LOCAL_MODEL_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Model cached locally.")
        return True
    return False

def run_local_inference_and_training():
    if not LOCAL_MODEL_PATH.exists():
        return
    
    # 1. Execute Native Inference
    print("Bootstrapping ExecuTorch native runtime engine...")
    et_runtime = Runtime.get()
    program = et_runtime.load_program(LOCAL_MODEL_PATH)
    forward_method = program.load_method("forward")
    
    mock_sensor_input = (torch.ones(1, 10),)
    outputs = forward_method.execute(mock_sensor_input)
    print(f"Native Forward Inference Complete. Outputs: {outputs[0]}")
    
    # 2. Simulate Local Backpropagation training step
    print("Running local training/gradients update simulation...")
    local_model = FederatedGlobalModel()
    
    # Simulate slightly modified weights to represent local sensor learning data
    local_state = local_model.state_dict()
    for key in local_state.keys():
        local_state[key] += torch.randn_like(local_state[key]) * 0.05
    local_model.load_state_dict(local_state)
    
    # 3. Serialize weights and POST up to the Central Node
    print("Shipping updated weights to central aggregator server...")
    buffer = io.BytesIO()
    torch.save(local_model.state_dict(), buffer)
    buffer.seek(0)
    
    files = {'file': ('weights.pt', buffer, 'application/octet-stream')}
    response = requests.post(f"{SERVER_BASE_URL}/upload_weights", files=files)
    print(f"Server Response: {response.json()}\n")

if __name__ == "__main__":
    if download_global_model():
        run_local_inference_and_training()
# import os
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI(title="Federated Learning Central Server")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# MODEL_DIR = "compiled_models"
# MODEL_FILE = "global_model.pte"

# @app.get("/")
# def read_root():
#     return {"status": "online", "component": "Federated Learning Server"}

# @app.get("/download_model")
# def download_model():
#     """
#     Endpoint for edge clients (FLient) to download the latest 
#     compiled ExecuTorch global model binary.
#     """
#     file_path = os.path.join(MODEL_DIR, MODEL_FILE)
    
#     if not os.path.exists(file_path):
#         raise HTTPException(status_code=404, detail="Global model binary not found. Run export script first.")
        
#     print(f"📤 Serving {MODEL_FILE} to edge client...")
#     return FileResponse(
#         path=file_path, 
#         filename=MODEL_FILE, 
#         media_type="application/octet-stream"
#     )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8001)

import os
import io
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from export_global_model import FederatedGlobalModel, compile_model_instance
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="Federated Learning Central Server")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


MODEL_DIR = "compiled_models"
MODEL_FILE = "global_model.pte"

# In-memory registry to hold local client updates for the current round
collected_updates = []
REQUIRED_CLIENTS = 2  # Set to 2 for quick local testing between devices

# Initialize the primary in-memory global weights registry
global_model = FederatedGlobalModel()

@app.get("/")
def read_root():
    return {
        "status": "online", 
        "round_status": f"{len(collected_updates)}/{REQUIRED_CLIENTS} clients checked-in"
    }

@app.get("/download_model")
def download_model():
    file_path = os.path.join(MODEL_DIR, MODEL_FILE)
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Global model binary not found.")
    return FileResponse(path=file_path, filename=MODEL_FILE, media_type="application/octet-stream")

@app.post("/upload_weights")
async def upload_weights(file: UploadFile = File(...)):
    """
    Accepts serialized client weights, parses the state dict, 
    and checks if an aggregation cycle should execute.
    """
    global collected_updates
    try:
        # Read the uploaded binary byte stream directly into memory
        file_bytes = await file.read()
        client_state_dict = torch.load(io.BytesIO(file_bytes), map_location="cpu")
        
        collected_updates.append(client_state_dict)
        print(f"Received weights update. Pool: {len(collected_updates)}/{REQUIRED_CLIENTS}")
        
        # Trigger Federated Averaging (FedAvg) if threshold is satisfied
        if len(collected_updates) >= REQUIRED_CLIENTS:
            print("Threshold met. Running FedAvg aggregation loop...")
            run_fedavg_and_recompile()
            return {"status": "success", "message": "Round complete. Global model compiled!"}
            
        return {"status": "success", "message": f"Weights registered. Awaiting {REQUIRED_CLIENTS - len(collected_updates)} more updates."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed processing weights file: {str(e)}")

def run_fedavg_and_recompile():
    global collected_updates, global_model
    
    # Extract keys and compute layer averages
    global_state = global_model.state_dict()
    for key in global_state.keys():
        layer_tensors = [client_update[key].float() for client_update in collected_updates]
        # Mathematical weight averaging across tensors
        global_state[key] = torch.stack(layer_tensors).mean(dim=0).to(global_state[key].dtype)
        
    # Reload newly aggregated weights into core model instance
    global_model.load_state_dict(global_state)
    print("Central model weights successfully aggregated.")
    
    # Flush current round weights storage registry
    collected_updates = []
    
    # Recompile the updated model down to an optimized .pte binary layout
    compile_model_instance(global_model, output_filename=MODEL_FILE)
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=os.getenv("TAILSCALE_IP"), port=8001)

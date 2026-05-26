from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from app.aggregator import FedAvgAggregator
import pydantic
from typing import Dict, List, Any
import uvicorn

app = FastAPI(title="Federated learning Core Orchestrator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

aggregator = FedAvgAggregator()

class WeightPayload(pydantic.BaseModel):
    client_id: str
    weights: Dict[str, List[float]]

@app.get("/model/global")
async def get_global_weights():
    """Returns flat representations of the global weight matrix."""
    return {
        "round": aggregator.global_round,
        "weights": aggregator.get_serializable_global_weights()
    }

@app.post("/model/upload")
async def upload_client_weights(payload: WeightPayload, background_tasks: BackgroundTasks):
    """Handles asynchronous weight ingestions via REST API."""
    aggregator.register_update(payload.client_id, payload.weights)
    if aggregator.can_aggregate(target_count=3):
        background_tasks.add_task(aggregator.aggregate)
        return {"status": "batch_received", "action": "aggregating"}
    return {"status": "buffered", "current_pool": len(aggregator.client_updates)}

@app.websocket("/ws/fl")
async def websocket_fl_stream(websocket: WebSocket):
    """WebSocket channel for duplex streaming of parameters."""
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "CLIENT_UPDATE":
                client_id = data.get("client_id")
                weights = data.get("weights")
                aggregator.register_update(client_id, weights)
                
                if aggregator.can_aggregate(target_count=3):
                    new_global = aggregator.aggregate()
                    await websocket.send_json({
                        "type": "ROUND_COMPLETE",
                        "round": aggregator.global_round,
                        "weights": new_global
                    })
    except WebSocketDisconnect:
        pass
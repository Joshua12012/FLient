import numpy as np
import torch
import logging
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FedAvgAggregator")

class FedAvgAggregator:
    def __init__(self):
        # Define structural layout of our Tiny CNN layers
        self.layer_shapes = {
            "conv1.weight": (8, 1, 3, 3),
            "conv1.bias": (8,),
            "conv2.weight": (16, 8, 3, 3),
            "conv2.bias": (16,),
            "fc1.weight": (10, 16 * 5 * 5),
            "fc1.bias": (10,)
        }
        self.global_round = 0
        self.global_weights: Dict[str, np.ndarray] = {}
        self.client_updates: Dict[str, Dict[str, np.ndarray]] = {}
        self.initialize_random_weights()

    def initialize_random_weights(self):
        """Initializes weight tensors using standard Xavier/Glorot uniform distributions."""
        logger.info("Initializing global weights matrix...")
        for layer, shape in self.layer_shapes.items():
            if "weight" in layer:
                # Basic Xavier uniform approximation
                bound = 1.0 / np.sqrt(shape[1] if len(shape) > 1 else shape[0])
                self.global_weights[layer] = np.random.uniform(-bound, bound, shape).astype(np.float32)
            else:
                self.global_weights[layer] = np.zeros(shape, dtype=np.float32)

    def register_update(self, client_id: str, payload: Dict[str, Any]):
        """Decodes incoming client flat lists back into structured layer arrays."""
        logger.info(f"Received weight updates from client: {client_id}")
        structured_weights = {}
        for layer, shape in self.layer_shapes.items():
            if layer in payload:
                structured_weights[layer] = np.array(payload[layer], dtype=np.float32).reshape(shape)
        self.client_updates[client_id] = structured_weights

    def can_aggregate(self, target_count: int = 3) -> bool:
        return len(self.client_updates) >= target_count

    def aggregate(self) -> Dict[str, List[float]]:
        """Executes the mathematical FedAvg computation across all collected client models."""
        if not self.client_updates:
            return {k: v.tolist() for k, v in self.global_weights.items()}

        logger.info(f"Executing aggregation round {self.global_round + 1}...")
        clients = list(self.client_updates.keys())
        
        for layer in self.layer_shapes.keys():
            # Extract layer tensor across all clients
            layer_tensors = [self.client_updates[cid][layer] for cid in clients]
            # Compute element-wise arithmetic mean
            self.global_weights[layer] = np.mean(layer_tensors, axis=0)

        self.global_round += 1
        self.client_updates.clear() # Reset buffering queue
        logger.info(f"Global round {self.global_round} finalized.")
        
        return self.get_serializable_global_weights()

    def get_serializable_global_weights(self) -> Dict[str, List[float]]:
        return {k: v.flatten().tolist() for k, v in self.global_weights.items()}
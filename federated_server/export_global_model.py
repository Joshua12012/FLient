import os
import torch
import torch.nn as nn
from torch.export import export
from executorch.exir import to_edge

# model architecture
class FederatedGlobalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 4)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# compiles the instance of .pt model weights into executorch binary
def compile_model_instance(model_instance, output_filename="global_model.pte"):
    model_instance.eval()
    example_input = (torch.randn(1, 10),)
    
    print("Tracing ATen graph topology...")
    captured_graph = export(model_instance, example_input)
    
    print("Lowering to executorch edge dialect")
    edge_program = to_edge(captured_graph)
    
    print("Serializing to optimize bytecode flatbuffer")
    executorch_program = edge_program.to_executorch()
    
    os.makedirs("compiled_models", exist_ok=True)
    target_path = os.path.join("compiled_models", output_filename)
    with open(target_path, "wb") as f:
        f.write(executorch_program.buffer)
    print(f"Global model compiled successfully at: {target_path}\n")

if __name__ == "__main__":
    # Setup initial baseline model weights
    initial_model = FederatedGlobalModel()
    compile_model_instance(initial_model)


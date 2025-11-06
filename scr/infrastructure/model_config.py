import modal
import os
from typing import Optional


MODEL_CONFIGS = {
    "phi-3.5-mini": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "gpu": modal.gpu.T4(),
        "gpu_memory_utilization": 0.75,
    },
    "llama-3.2-1b": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "gpu": modal.gpu.T4(),
        "gpu_memory_utilization": 0.75,
    },
    "llama-3.2-3b": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "gpu": modal.gpu.A100(count=1),
        "gpu_memory_utilization": 0.85,
    },
    # Mistral additions
    "mistral-7b-instruct-v0.3": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",  
        "gpu": modal.gpu.A100(count=1),                 
        "gpu_memory_utilization": 0.85,
    },
    "mixtral-8x7b-instruct-v0.1": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",  
        "gpu": modal.gpu.A100(count=1),                    
        "gpu_memory_utilization": 0.8,
    },
}



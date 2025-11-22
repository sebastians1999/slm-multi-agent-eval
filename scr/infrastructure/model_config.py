MODEL_CONFIGS = {
    
    "Qwen/Qwen3-4B-Instruct-2507":{
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "gpu": "T4",
        "max_model_len": 10000,
        "tool_parser": "hermes",
        "artificial_analysis_name": "Qwen3 4B 2507 (Reasoning)",
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "gpu": "T4",
        "max_model_len": 8000,
        "tool_parser": "hermes",
    },
    "meta-llama/Llama-3.1-8B": {
        "model_id": "meta-llama/Llama-3.1-8B",
        "gpu": "T4",
        "max_model_len": 8192,
        "tool_parser": "hermes"
    },
    "microsoft/Phi-4-mini-instruct": {
        "model_id": "microsoft/Phi-4-mini-instruct",
        "gpu": "L4",
        "max_model_len": 8000,
        "tool_parser": "hermes",
    },
    "meta-llama/Llama-3.2-1B-Instruct": {
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "gpu": "T4",
        "max_model_len": 8192,
        "tool_parser": "llama3_json",
    },
    "meta-llama/Llama-3.2-3B-Instruct": {
        "model_id": "meta-llama/Llama-3.2-3B-Instruct",
        "gpu": "A100",
        "max_model_len": 32768,
        "tool_parser": "llama3_json",
    },
    "mistralai/Mistral-7B-Instruct-v0.3": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.3",
        "gpu": "L4",
        "max_model_len": 10000,
        "tool_parser": "mistral",
    },
    "mistralai/Mixtral-8x7B-Instruct-v0.1": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "gpu": "A100",
        "max_model_len": 32768,
        "tool_parser": "mistral",
    },
}
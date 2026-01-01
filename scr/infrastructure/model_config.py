MODEL_CONFIGS = {
    
    "Qwen/Qwen3-4B-Instruct-2507":{
        "model_id": "Qwen/Qwen3-4B-Instruct-2507",
        "gpu": "L4",
        "max_model_len": 30000,
        "tool_parser": "hermes",
        "artificial_analysis_name": "Qwen3 4B 2507 (Reasoning)"
    },
    "Qwen/Qwen3-8B": {
        "model_id": "Qwen/Qwen3-8B",
        "gpu": "A100",             # Recommend A100 for long context; can drop to L4 if you cap context
        "max_model_len": 32768,    # Safe cap for L4/A10; set 128000 if serving on A100 with enough memory
        "tool_parser": "hermes",
        "artificial_analysis_name": "Qwen3 8B (Reasoning)"      
    },
    "gemini-1.5-flash-8b": {
        "model_id": "gemini-1.5-flash-8b",
        "gpu": None,                # Hosted model; not run via vLLM/Modal
        "max_model_len": None,      # Use provider default
        "tool_parser": None,
        "artificial_analysis_name": "Gemini 1.5 Flash 8B"
    },
    "google/gemma-3-4b-it": {
        "model_id": "google/gemma-3-4b-it",
        "gpu": "L4",
        "max_model_len": 8192,
        "tool_parser": None,
        "artificial_analysis_name": "Gemma 3 4B Instruct",
    },
    "microsoft/Phi-3.5-mini-instruct": {
        "model_id": "microsoft/Phi-3.5-mini-instruct",
        "gpu": "T4",
        "max_model_len": 8000,
        "tool_parser": "hermes",
    },
    "microsoft/Phi-3-mini-4k-instruct": {
        "model_id": "microsoft/Phi-3-mini-4k-instruct",
        "gpu": "T4",
        "max_model_len": 4096,
        "tool_parser": "hermes",
        "artificial_analysis_name": "Phi-3 Mini Instruct 3.8B",
    },
    "meta-llama/Llama-3.1-8B": {
        "model_id": "meta-llama/Llama-3.1-8B",
        "gpu": "A100",               # Better headroom than T4; use A100 if available
        "max_model_len": 8192,     # Raise cap to avoid prompt overflow while staying safe on L4
        "tool_parser": "llama3_json",
        "chat_template": """{{ bos_token }}{% for message in messages %}<|start_header_id|>{{ message['role'] }}<|end_header_id|>
{% if message['content'] is string %}{{ message['content'] }}{% else %}{% for item in message['content'] %}{% if item['type'] == 'text' %}{{ item['text'] }}{% endif %}{% endfor %}{% endif %}<|eot_id|>
{% endfor %}<|start_header_id|>assistant<|end_header_id|>
""",
        "artificial_analysis_name": "Llama 3.1 8B"
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
    "meta-llama/Llama-3.1-70B-Instruct": {
        "model_id": "meta-llama/Llama-3.1-70B-Instruct",
        "gpu": "A100",
        "max_model_len": 8192,
        "tool_parser": "llama3_json",
        "artificial_analysis_name": "Llama 3.1 70B Instruct"
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        "model_id": "meta-llama/Llama-3.3-70B-Instruct",
        "gpu": "A100",
        "max_model_len": 8192,
        "tool_parser": "llama3_json",
        "artificial_analysis_name": "Llama 3.3 70B Instruct"
    },
    "Qwen/Qwen2.5-72B-Instruct": {
        "model_id": "Qwen/Qwen2.5-72B-Instruct",
        "gpu": "A100",
        "max_model_len": 32768,
        "tool_parser": "hermes",
        "artificial_analysis_name": "Qwen2.5 72B Instruct"
    },
    "Qwen/Qwen2.5-7B-Instruct": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "gpu": "L4",
        "max_model_len": 32768,
        "tool_parser": "hermes",
        "artificial_analysis_name": "Qwen2.5 7B Instruct"
    },
    "mistralai/Ministral-8B-Instruct-2410": {
        "model_id": "mistralai/Ministral-8B-Instruct-2410",
        "gpu": "L4",               
        "max_model_len": 32768,    
        "tool_parser": "mistral",
        "artificial_analysis_name": "Ministral 8B",
    },
    "Qwen/Qwen3-32B-AWQ": {
        "model_id": "Qwen/Qwen3-32B-AWQ",
        "gpu": "A100",
        "max_model_len": 8192,
        "tool_parser": "hermes",
        "artificial_analysis_name": "Qwen3 32B (Reasoning)"
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8": {
        "model_id": "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8",
        "gpu": "A100",
        "max_model_len": 8192,
        "tool_parser": "hermes",
        "artificial_analysis_name": "Qwen3 30B A3B (Non-reasoning)"
    },
}

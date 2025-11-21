# Serving vLLM on Modal

This guide walks through the full workflow for running `scr/infrastructure/modal_llm_server.py` on [Modal](https://modal.com): initial CLI setup, configuring per-model metadata, and launching the server so the evaluation pipeline can call it.

## 1. One-time Modal setup

1. Install the CLI: `pip install modal` (already included when you run `uv sync`).
2. Authenticate: `modal token new` and finish the browser flow.

## 2. Secrets and environment variables

Two secrets are required by the server:

- **Hugging Face access** – create once:  
	`modal secret create huggingface-secret HF_TOKEN=hf_...`
- **Runtime model selection** – the file uses `modal.Secret.from_local_environ(env_keys=["MODEL_ID"])`, so the `MODEL_ID` env var that you set locally is injected into the container. Example invocation:  
	``MODEL_ID="Qwen/Qwen3-4B-Instruct-2507" modal serve scr/infrastructure/modal_llm_server.py``


## 3. Configure `MODEL_CONFIGS`

`scr/infrastructure/model_config.py` stores per-model settings that the server references at import time:

- `gpu`: GPU tier Modal should provision. Choose one whose compute capability matches the kernels you need (e.g., FlexAttention v2 requires ≥ SM 8.0, so T4s will fail for some models).
- `max_model_len`: Optional cap on context length. Supplying this keeps vLLM from reserving more KV cache than the GPU can handle; leave it `None` to let vLLM auto-detect.
- `tool_parser`: Chat template/langchain parser to use (e.g., `mistral`, `llama3_json`).
- `artificial_analysis_name`: Some models have different names on [Artificial Analysis](https://artificialanalysis.ai) compared with Hugging Face. The cost tracker (`scr/utilities/cost_tracker.py`) checks this field so it can map the Hugging Face slug to the correct pricing slug before calling the API. Populate it whenever the names diverge to keep the cost numbers accurate.

```python
MODEL_CONFIGS = {
		"Qwen/Qwen3-4B-Instruct-2507": {
				"gpu": "L4",
				"max_model_len": 30000,
				"tool_parser": "hermes",
				"artificial_analysis_name": "Qwen3 4B 2507 (Reasoning)",
		},
		# ... other entries ...
}
```

## 4. Running the server

```bash
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.3" modal serve scr/infrastructure/modal_llm_server.py
# or switch MODEL_ID to any other key present in MODEL_CONFIGS
```

During `VLLMServer.initialize()`:

- vLLM builds `AsyncEngineArgs`, dynamically attaching `max_model_len` only when the config provides it and switching to `attention_backend="FLASH_ATTN"` on GPUs (like T4s) that cannot run FlexAttention v2.
- Hugging Face weights download into `/models`, backed by the shared Modal volume for cache reuse.
- `ZeusMonitor` tracks GPU energy usage so `/v1/chat/completions` responses include `energy_consumption` plus the downstream Artificial Analysis cost breakdown.

**Modal prints a public HTTPS endpoint such as `https://<user>--slm-server-vllmserver-serve-dev.modal.run/v1`. Point `Eval_pipeline` (see `main.py`) at this base URL. Use this as the base_url in the main. Also make sure to use the same model name as being served!**

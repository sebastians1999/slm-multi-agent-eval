
MODEL_ID="meta-llama/Llama-3.2-1B-Instruct" GPU_TYPE="T4" modal serve scr/infrastructure/modal_llm_server.py

MODEL_ID="meta-llama/Llama-3.2-3B-Instruct" GPU_TYPE="A100" modal serve scr/infrastructure/modal_llm_server.py

MODEL_ID="microsoft/Phi-3.5-mini-instruct" GPU_TYPE="L4" modal serve scr/infrastructure/modal_llm_server.py



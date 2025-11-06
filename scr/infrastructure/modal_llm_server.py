import modal
import os

from modal.stream_type import StreamType


GPU_TYPE = os.environ.get("GPU_TYPE", "T4")
DEFAULT_MODEL_ID = "microsoft/Phi-3.5-mini-instruct"

MINUTES = 60


vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.8.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm==0.11.0",
        "huggingface_hub[hf_transfer]==0.35.0",
        "zeus-ml",
        "fastapi[standard]"
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",  # faster model transfers
    })
)


app = modal.App("slm-server")
volume = modal.Volume.from_name("model-cache", create_if_missing=True)


runtime_config = modal.Secret.from_local_environ(
    env_keys=["MODEL_ID","GPU_TYPE"],  
)

runtime_secrets = modal.Secret.from_name("huggingface-secret")


@app.cls(
    image=vllm_image,
    gpu=GPU_TYPE,  # Read at import time from local env
    scaledown_window=15 * MINUTES,  # how long should we stay up with no requests?
    timeout=10 * MINUTES,  # how long should we wait for container start?
    volumes={"/models": volume},
    secrets=[runtime_config, runtime_secrets],  # Inject runtime config into container
)
@modal.concurrent(  # how many requests can one replica handle? tune carefully!
    max_inputs=32
)
class VLLMServer:
    @modal.enter()
    async def initialize(self):
        """Initialize vLLM engine and Zeus monitor on container startup."""
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
        from zeus.monitor import ZeusMonitor
        import torch

        
        MODEL_ID = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        
        hf_token = os.environ.get("HF_TOKEN")
        
        if hf_token:
            print("✓ Huggingface token is set.")

        print(f"✓ Initializing with MODEL_ID: {MODEL_ID}")
        print(f"✓ GPU Type: {GPU_TYPE}")

    
        self.monitor = ZeusMonitor(gpu_indices=[torch.cuda.current_device()])


        engine_args = AsyncEngineArgs(
            model=MODEL_ID,
            dtype="float16" if GPU_TYPE == "T4" else "auto",
            download_dir="/models",
            enforce_eager=True,
            max_model_len=8192,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        model_config = await self.engine.get_model_config()

        
        base_model = BaseModelPath(name=MODEL_ID, model_path=MODEL_ID)

       
        serving_models = OpenAIServingModels(
            model_config=model_config,
            engine_client=self.engine,
            base_model_paths=[base_model],
            lora_modules=None,
        )

        self.chat_handler = OpenAIServingChat(
            engine_client=self.engine,
            model_config=model_config,
            models=serving_models,
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )

        print(f"✓ Server ready: {MODEL_ID} on {GPU_TYPE}")

    @modal.exit()
    def cleanup(self):
            self.engine.shutdown()

    @modal.asgi_app()
    def serve(self):
        """Create and return FastAPI app."""
        from vllm.entrypoints.openai.protocol import ChatCompletionRequest
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        import time
        import uuid

        app = FastAPI()

        @app.get("/v1/models")
        async def models():
            """LangChain needs this endpoint."""
            return {
                "data": [
                    {
                        "id": MODEL_ID,
                        "object": "model",
                        "owned_by": "vllm",
                    }
                ]
            }

        @app.post("/v1/chat/completions")
        async def chat_completions(request: Request):
            """OpenAI-compatible endpoint with energy tracking."""
            raw_request = await request.json()

           
            chat_request = ChatCompletionRequest(**raw_request)

            request_id = str(uuid.uuid4())[:8]

            # Start tracking
            self.monitor.begin_window(f"req_{request_id}")
            start_time = time.time()

            try:
                # Generate response
                response = await self.chat_handler.create_chat_completion(chat_request)

                measurement = self.monitor.end_window(f"req_{request_id}")
                duration = time.time() - start_time

                # Convert to dict and add energy
                response_dict = response.model_dump()  # serialize to dictionary
                response_dict["energy_consumption"] = {
                    "joules": measurement.total_energy,
                    "duration_seconds": duration,
                    "watts": measurement.total_energy / duration,
                }

                return JSONResponse(response_dict)

            except Exception as e:
                import traceback
                error_trace = traceback.format_exc()
                print(f"Error in chat_completions: {error_trace}")
                try:
                    self.monitor.end_window(f"req_{request_id}")
                except:
                    pass
                return JSONResponse({"error": str(e), "traceback": error_trace}, status_code=500)

        return app

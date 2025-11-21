from datasets import load_dataset
from scr.pipeline.eval_pipeline import Eval_pipeline

def main():
    dataset = load_dataset("gsm8k", "main")["test"].select(range(1))  # sample 50 items

    pipeline = Eval_pipeline(
        dataset=dataset,
        model="Qwen/Qwen3-4B-Instruct-2507",
        #base_url="ht"tp://localhost:11434/v1",
        base_url= "https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1",
    )

    pipeline.run_eval()

if __name__ == "__main__":
    main()

from datasets import load_dataset
from scr.pipeline.eval_pipeline import Eval_pipeline

def main():
    dataset = load_dataset("gsm8k", "main")["test"].select(range(100))  # sample 50 items

    pipeline = Eval_pipeline(
        dataset=dataset,
        model="llama3.1:8b",
        base_url="http://localhost:11434/v1",
    )

    pipeline.run_eval()

if __name__ == "__main__":
    main()

from scr.pipeline.eval_pipeline import Eval_pipeline
import os
from dotenv import load_dotenv
from datasets import load_dataset


def main():

    load_dotenv()
    base_url = os.getenv("MODAL_BASE_URL")
    api_key = "EMPTY"
    
    
    #Settings 
    model = "microsoft/Phi-3.5-mini-instruct"
    temperature = 0.3
    use_web_search = True
    max_iterations = 2
    


    print("\nLoading GAIA benchmark dataset...")
    dataset = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        trust_remote_code=True
    )

    test_data = dataset["test"]
    validation_data = dataset["validation"]

    print(f"  Test samples: {len(test_data)}")
    print(f"  Validation samples: {len(validation_data)}")

    eval_data = validation_data.select(range(1))
    print(f"\n→ Evaluating on {len(eval_data)} validation samples")

    pipeline = Eval_pipeline(
        dataset=eval_data,
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        max_iterations=max_iterations,
        use_web_search=use_web_search,
    )

    print("\nStarting evaluation")
    pipeline.run_eval()

    print("Evaluation completed!")



if __name__ == "__main__":
    main()

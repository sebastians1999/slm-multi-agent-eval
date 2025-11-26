from scr.pipeline.GAIA_benchmark.eval_pipeline import Eval_pipeline
from scr.agents.single_agent.single_agent import SingleAgent
import os
from dotenv import load_dotenv
from datasets import load_dataset


def main():

    load_dotenv()
    base_url = os.getenv("MODAL_BASE_URL")
    api_key = "EMPTY"


    # Agent settings
    #model = "Qwen/Qwen3-4B-Instruct-2507"
    model = "Qwen/Qwen3-32B-AWQ"
    temperature = 0.2
    max_iterations = 5



    print("\nLoading GAIA benchmark dataset...")
    dataset = load_dataset(
        "gaia-benchmark/GAIA",
        "2023_all",
        trust_remote_code=True
    )

    test_data = dataset["test"]
    validation_data = dataset["validation"]

    filtered_validation_data = validation_data.filter(lambda example: example["file_name"]== "")


    print(f"  Test samples: {len(test_data)}")
    print(f"  Validation samples: {len(validation_data)}")
    print(f"(Filtered to only include samples with empty file_name (no documents)):{len(filtered_validation_data)} samples")
    #eval_data = filtered_validation_data.select(range(1))
    eval_data = filtered_validation_data
    print(f"\n→ Evaluating on {len(eval_data)} validation samples")

    agent = SingleAgent(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
        max_iterations=max_iterations,
        tool_categories=['search', 'code', 'browser']  
    )

    pipeline = Eval_pipeline(
        dataset=eval_data,
        agent=agent
    )

    # 3. Run evaluation
    print("\nStarting evaluation")
    pipeline.run_eval()

    print("Evaluation completed!")



if __name__ == "__main__":
    main()

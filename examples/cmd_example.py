"""
Example: Using CMD (Conquer-and-Merge Discussion) Framework with GAIA Benchmark

This example demonstrates how to evaluate the CMD multi-agent architecture
on the GAIA benchmark dataset, similar to main.py but with collaborative discussion.
"""

from scr.agents.cmd import AgentConfig, AgentGroup, CMDOrchestrator
from scr.pipeline.GAIA_benchmark.eval_pipeline import Eval_pipeline
import os
from dotenv import load_dotenv
from datasets import load_dataset


def main():
    # Load environment variables
    load_dotenv()
    base_url = os.getenv("MODAL_BASE_URL")
    api_key = "EMPTY"

    # Agent settings
    model = "Qwen/Qwen3-4B-Instruct-2507"
    temperature = 0.3
    max_iterations = 5

    # Define custom agent configurations for CMD
    custom_configs = [
        AgentConfig(
            agent_id="searcher",
            model_name=model,
            system_prompt="Focus on finding factual information using search and web browsing. Always verify facts before presenting them.",
            tool_categories=["search", "browser"],
            temperature=temperature,
            max_iterations=max_iterations
        ),
        AgentConfig(
            agent_id="coder",
            model_name=model,
            system_prompt="Solve problems with code execution. Break down complex problems systematically and use computation when needed.",
            tool_categories=["code"],
            temperature=temperature,
            max_iterations=max_iterations
        ),
        AgentConfig(
            agent_id="generalist",
            model_name=model,
            system_prompt="Think broadly and consider all available tools. Provide balanced perspectives and use the right tool for each task.",
            tool_categories=["search", "code", "browser"],
            temperature=temperature,
            max_iterations=max_iterations
        ),
    ]

    # Load GAIA benchmark dataset
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
    print(f"\n→ Evaluating on {len(eval_data)} validation samples with CMD")

    # 1. Create agent group
    print("\nCreating CMD agent group with 3 agents...")
    agent_group = AgentGroup(custom_configs)
    print(f"  Agent IDs: {agent_group.agent_ids}")

    # 2. Create CMD orchestrator
    orchestrator = CMDOrchestrator(
        agent_group=agent_group,
        max_rounds=2,  # 2 rounds of discussion
        enable_secretary=True  # Use secretary for tie-breaking
    )

    # 3. Initialize eval pipeline with CMD orchestrator
    pipeline = Eval_pipeline(
        dataset=eval_data,
        agent=orchestrator  # CMDOrchestrator has run() method compatible with eval_pipeline
    )

    # 4. Run evaluation
    print("\nStarting CMD evaluation")
    pipeline.run_eval()

    print("\nCMD Evaluation completed!")


if __name__ == "__main__":
    main()

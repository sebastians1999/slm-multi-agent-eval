from datasets import load_dataset
from scr.pipeline.eval_pipeline import Eval_pipeline

# Import agents for comparison
from scr.agents.agent_01.multiAgent import MultiAgent
from scr.agents.agent_02.singleAgent import SingleAgent

def main():
    # Load dataset
    dataset = load_dataset("gsm8k", "main")["test"].select(range(200))  # Use a subset for quicker evaluation

    agent_type = "single" 

    if agent_type == "multi":
        print("\n" + "="*80)
        print("Running MULTI-AGENT evaluation (SLM with specialized agents)")
        print("="*80 + "\n")

        agent = MultiAgent(
            model="microsoft/Phi-3-mini-4k-instruct",
            temperature=0.3,
            base_url="https://francescomoscardelli1--slm-server-vllmserver-serve-dev.modal.run/v1",
            api_key="",
            max_iterations=2,
            use_web_search=False,
        )

    elif agent_type == "single":
        print("\n" + "="*80)
        print("Running SINGLE-AGENT evaluation (Large LLM)")
        print("="*80 + "\n")

        agent = SingleAgent(
            model="Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", 
            temperature=0.3,
            base_url="https://francescomoscardelli1--slm-server-vllmserver-serve-dev.modal.run/v1",
            api_key="",
        )

    else:
        raise ValueError(f"Invalid agent_type: {agent_type}. Use 'multi' or 'single'.")

    pipeline = Eval_pipeline(
        dataset=dataset,
        model=agent.model,  
        temperature=agent.temperature,
        base_url=agent.base_url,
        api_key=agent.api_key,
        max_iterations=getattr(agent, 'max_iterations', 1),
        use_web_search=getattr(agent, 'use_web_search', False),
        agent=agent,
    )

    pipeline.run_eval()

if __name__ == "__main__":
    main()

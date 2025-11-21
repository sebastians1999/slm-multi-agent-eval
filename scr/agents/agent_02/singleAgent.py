from __future__ import annotations
from typing import Dict, Any, Optional
import uuid
from datetime import datetime

from scr.agents.base_agent import AgentMetaData, BaseMultiAgent, StructuredOutput
from scr.utilities.helper_functions import extract_final_answer
from scr.utilities.prompts import GENERAL_PROMPT



class SingleAgent(BaseMultiAgent):
    """
    Single-agent LLM solver for benchmarking.

    This is a simple baseline that:
    1. Takes a problem
    2. Sends it to an LLM with direct prompting
    3. Returns the solution with metadata tracking

    Designed for comparison against multi-agent approaches.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.3,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        max_iterations: int = 1,  # Not used, but kept for interface compatibility
        use_web_search: bool = False,  # Not used, but kept for interface compatibility
        **kwargs
    ):
        """
        Initialize the single agent.

        Args:
            model: Model name/identifier
            temperature: Sampling temperature
            base_url: OpenAI-compatible API endpoint
            api_key: API key for authentication
            max_iterations: Not used (kept for compatibility with eval_pipeline)
            use_web_search: Not used (kept for compatibility with eval_pipeline)
            **kwargs: Additional configuration
        """
        super().__init__(model, temperature, base_url, api_key, **kwargs)


    def run(self, problem: str) -> Dict[str, Any]:
        """
        Run the single agent on a problem.

        Args:
            problem: The question/problem to solve

        Returns:
            Dict containing:
                - solution: Full response text
                - structured_output: StructuredOutput with model_answer and reasoning_trace
                - metadata: AgentMetaData with tokens, energy, timing info
        """
        session_id = f"sess-{uuid.uuid4().hex[:8]}"

        # Initialize metadata
        metadata = AgentMetaData(
            session_id=session_id,
            start_time=datetime.now(),
            model_name=self.model,
            temperature=self.temperature,
            base_url=self.base_url,
        )

        # Construct the prompt
        prompt = f"{GENERAL_PROMPT}\n\nQuestion: {problem}"


        # Call the LLM
        messages = [{"role": "user", "content": prompt}]
        response_dict = self.invoke(messages)

        # Extract the solution text
        solution = response_dict["choices"][0]["message"]["content"]


        # Extract usage metadata
        usage = response_dict.get("usage", {})
        metadata.completion_tokens = usage.get("completion_tokens", 0)
        metadata.prompt_tokens = usage.get("prompt_tokens", 0)
        metadata.total_tokens = usage.get("total_tokens", 0)
        metadata.iterations = 1  # Single call

        energy_data = response_dict.get("energy_consumption", {})
        if energy_data:
            metadata.total_energy_joules = energy_data.get("joules", 0.0)
            metadata.total_duration_seconds = energy_data.get("duration_seconds", 0.0)

            if metadata.total_duration_seconds > 0:
                metadata.average_watts = metadata.total_energy_joules / metadata.total_duration_seconds


        metadata.end_time = datetime.now()
        model_answer = extract_final_answer(solution)
        structured_output = StructuredOutput(
            model_answer=model_answer,
            reasoning_trace=solution
        )


        # Return in the format expected by eval_pipeline
        return {
            "solution": solution,
            "structured_output": structured_output,
            "metadata": metadata,
            "session_id": session_id,
        }




def run_single_agent(
    problem: str,
    model: str,
    temperature: float = 0.3,
    base_url: Optional[str] = None,
    api_key: str = "EMPTY",
) -> Dict[str, Any]:
    """
    Convenience function to run the single agent.

    Args:
        problem: The question/problem to solve
        model: Model name
        temperature: Sampling temperature
        base_url: OpenAI-compatible base URL
        api_key: API key

    Returns:
        Result dict with solution and metadata
    """
    agent = SingleAgent(
        model=model,
        temperature=temperature,
        base_url=base_url,
        api_key=api_key,
    )
    return agent.run(problem)



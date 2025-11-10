from datasets import Dataset
from typing import Optional, List, Dict, Any
from datetime import datetime
import os
from pathlib import Path
import json
from tqdm import tqdm

# Import multi-agent
from scr.agents.agent_01.multiAgent import MultiAgent

# Import cost tracker
from scr.utilities.cost_tracker import CostTracker


class Eval_pipeline:
    """
    Evaluation pipeline for running multi-agent system on datasets.

    Captures structured outputs and metadata (tokens, energy, timing, etc.)
    """

    def __init__(
        self,
        dataset: Dataset,
        model: str,
        temperature: float = 0.3,
        base_url: Optional[str] = None,
        api_key: str = "EMPTY",
        max_iterations: int = 2,
        use_web_search: bool = True,
        log_folder_path: Optional[str] = None,
    ):
        """
        Initialize evaluation pipeline.

        Args:
            dataset: HuggingFace dataset to evaluate
            model: Model name/ID
            temperature: Sampling temperature
            base_url: OpenAI-compatible API base URL
            api_key: API key for the model
            max_iterations: Maximum feedback iterations per problem
            use_web_search: Whether to enable web search
            log_folder_path: Custom path for logs (default: repo/logs)
        """
        self.dataset = dataset
        self.model = model
        self.temperature = temperature
        self.base_url = base_url
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.use_web_search = use_web_search

        # Set up log folder
        if log_folder_path is None:
            self.log_folder_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                ),
                "logs",
            )
        else:
            self.log_folder_path = log_folder_path

        self.agent = MultiAgent(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            max_iterations=max_iterations,
            use_web_search=use_web_search,
        )

        # Initialize cost tracker
        self.cost_tracker = CostTracker()

    def run_eval(self):
        """
        Run evaluation on the entire dataset.

        Collects structured outputs, metadata, and saves logs.
        """
        logs = []
        logs_formatted = []

        for sample in tqdm(self.dataset, desc="Evaluating"):
            question = sample.get("Question")
            if not question:
                print(f"Skipping sample {sample.get('task_id')} - no question found")
                continue

            result = self.call_agent(question=question)

            structured_output = result.get("structured_output")
            metadata = result.get("metadata")

            cost_data = None
            if metadata and metadata.prompt_tokens and metadata.completion_tokens:
                cost_data = self.cost_tracker.calculate_cost(
                    model_name=self.model,
                    input_tokens=metadata.prompt_tokens,
                    output_tokens=metadata.completion_tokens
                )

            result_json = {
                "task_id": sample["task_id"],
                "model_answer": structured_output.model_answer if structured_output else None,
                "reasoning_trace": structured_output.reasoning_trace if structured_output else None,
            }

            log = {
                "task_id": sample["task_id"],
                "file_name": sample.get("file_name"),
                "file_path": sample.get("file_path"),
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "level": sample.get("Level"),
                "annotator_metadata": sample.get("Annotator Metadata"),
                "result": result_json,
                "agent_metadata": self._serialize_metadata(metadata) if metadata else None,
                "cost_data": cost_data,
                "model_temperature": self.temperature,
                "model": self.model
            }

            logs.append(log)
            logs_formatted.append(result_json)

        # Save logs
        self.save_logs(logs_list=logs, logs_list_formatted=logs_formatted)

        return logs

    def call_agent(self, question: str) -> Dict[str, Any]:
        """
        Call the agent to solve a single question.

        Args:
            question: The problem/question to solve

        Returns:
            Full state dict with structured_output and metadata
        """
        try:
            result = self.agent.run(problem=question)
            return result
        except Exception as e:
            print(f"Error running agent on question: {e}")
            return {
                "structured_output": None,
                "metadata": None,
                "error": str(e)
            }

    def _serialize_metadata(self, metadata) -> Dict[str, Any]:
        """Convert AgentMetaData to serializable dict."""
        if metadata is None:
            return None

        return {
            "session_id": metadata.session_id,
            "start_time": metadata.start_time.isoformat() if metadata.start_time else None,
            "end_time": metadata.end_time.isoformat() if metadata.end_time else None,
            "model_name": metadata.model_name,
            "temperature": metadata.temperature,
            "iterations": metadata.iterations,
            "completion_tokens": metadata.completion_tokens,
            "prompt_tokens": metadata.prompt_tokens,
            "total_tokens": metadata.total_tokens,
            "total_energy_joules": metadata.total_energy_joules,
            "total_duration_seconds": metadata.total_duration_seconds,
            "average_watts": metadata.average_watts,
        }

    def save_logs(
        self,
        logs_list: List[Dict[str, Any]],
        logs_list_formatted: List[Dict[str, Any]]
    ) -> None:
        """
        Save evaluation logs to JSON files.

        Args:
            logs_list: Full logs with metadata
            logs_list_formatted: Formatted results for submission
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = Path(self.log_folder_path) / f"eval_{timestamp}"
        os.makedirs(folder, exist_ok=True)


        file_path_logs = Path(folder) / "eval_logs.json"
        with open(file_path_logs, "w") as f:
            json.dump(logs_list, f, indent=2)


        file_path_logs_formatted = Path(folder) / "results.json"
        with open(file_path_logs_formatted, "w") as f:
            json.dump(logs_list_formatted, f, indent=2)

        print(f"\nLogs saved to: {folder}")

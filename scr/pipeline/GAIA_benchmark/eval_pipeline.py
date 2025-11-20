from datasets import Dataset
from typing import List, Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path
import json
from tqdm import tqdm

from scr.agents.graph_state import GraphState
from scr.utilities.cost_tracker import CostTracker


class Eval_pipeline:
    """
    Evaluation pipeline for running agents on datasets.

    Captures structured outputs and metadata (tokens, energy, timing, etc.)
    """

    def __init__(
        self,
        dataset: Dataset,
        agent,
        log_folder_path: Optional[str] = None,
    ):
        """
        Initialize evaluation pipeline.

        Args:
            dataset: HuggingFace dataset to evaluate
            agent: Agent instance with run(problem: str) -> GraphState method
            log_folder_path: Custom path for logs (default: repo/logs)
        """
        self.dataset = dataset
        self.agent = agent

        if log_folder_path is None:
            # Go up to project root: eval_pipeline.py -> GAIA_benchmark -> pipeline -> scr -> project_root
            self.log_folder_path = os.path.join(
                os.path.dirname(
                    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                ),
                "logs",
            )
        else:
            self.log_folder_path = log_folder_path

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

            if result is None:
                print(f"Skipping sample {sample.get('task_id')} - agent returned None")
                continue

            structured_output = result.structured_output

            cost_data = None
            
            if result.model_name == "Qwen3-4B-Instruct-2507":
                
                model = "Qwen3 4B"
                
            else: 
                
                model = result.model_name
            

            
            print(f"Calculating cost for :{result.model_name}")
            
            if result.total_prompt_tokens and result.total_completion_tokens:
                cost_data = self.cost_tracker.calculate_cost(
                    model_name=model,
                    input_tokens=result.total_prompt_tokens,
                    output_tokens=result.total_completion_tokens
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
                "agent_metadata": self._serialize_metadata(result),
                "cost_data": cost_data,
                "model_temperature": result.temperature,
                "model": result.model_name
            }

            logs.append(log)
            logs_formatted.append(result_json)

        self.save_logs(logs_list=logs, logs_list_formatted=logs_formatted)

        return logs

    def call_agent(self, question: str) -> Optional[GraphState]:
        """
        Call the agent to solve a single question.

        Args:
            question: The problem/question to solve

        Returns:
            GraphState with structured output and metadata, or None on error
        """
        try:
            result = self.agent.run(problem=question)
            return result
        except Exception as e:
            print(f"Error running agent on question: {e}")
            return None

    def _serialize_metadata(self, state: GraphState) -> Dict[str, Any]:
        """Convert GraphState metadata to serializable dict."""
        if state is None:
            return None

        return {
            "session_id": state.session_id,
            "start_time": state.start_time.isoformat() if state.start_time else None,
            "end_time": state.end_time.isoformat() if state.end_time else None,
            "model_name": state.model_name,
            "temperature": state.temperature,
            "iterations": state.iterations,
            "completion_tokens": state.total_completion_tokens,
            "prompt_tokens": state.total_prompt_tokens,
            "total_tokens": state.total_tokens,
            "total_energy_joules": state.total_energy_joules,
            "total_duration_seconds": state.total_duration_seconds,
            "average_watts": state.average_watts,
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

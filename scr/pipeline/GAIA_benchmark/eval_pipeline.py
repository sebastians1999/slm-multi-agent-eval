from datasets import Dataset
from typing import List, Dict, Any, Optional, TypedDict
from datetime import datetime
import os
from pathlib import Path
import json
from tqdm import tqdm
from .scorer import question_scorer

from scr.agents.graph_state import GraphState
from scr.utilities.cost_tracker import CostTracker
from scr.infrastructure.model_config import MODEL_CONFIGS


class ResultJson(TypedDict):
    """Structure for the result field."""
    task_id: str
    model_answer: Optional[str]
    reasoning_trace: Optional[str]


class AgentMetadata(TypedDict):
    """Structure for agent metadata."""
    session_id: str
    start_time: Optional[str]  
    end_time: Optional[str]    
    model_name: str
    temperature: float
    iterations: int
    completion_tokens: int
    prompt_tokens: int
    total_tokens: int
    total_energy_joules: float
    total_duration_seconds: float
    average_watts: float


class CostData(TypedDict):
    """Structure for cost data from CostTracker."""
    input_cost: float
    output_cost: float
    total_cost: float
    price_per_1m_input_tokens: float
    price_per_1m_output_tokens: float


class EvalLog(TypedDict):
    """Structure for evaluation log entries."""
    task_id: str
    file_name: Optional[str]
    file_path: Optional[str]
    timestamp: str
    question: str
    level: Optional[str]
    final_answer: Optional[str]
    annotator_metadata: Optional[Any]
    result: ResultJson
    score: bool
    agent_metadata: AgentMetadata
    cost_data: Optional[CostData]
    model_temperature: float
    model: str


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
        logs: List[EvalLog] = []
        logs_formatted: List[ResultJson] = []

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

            cost_data: Optional[CostData] = None

            # Map model name to cost tracking name if configured
            model = result.model_name
            if MODEL_CONFIGS.get(result.model_name, {}).get("artificial_analysis_name"):
                model = MODEL_CONFIGS[result.model_name]["artificial_analysis_name"]

            print(f"Calculating cost for: {result.model_name} (mapped to: {model})")
            
            if result.total_prompt_tokens and result.total_completion_tokens:
                cost_data = self.cost_tracker.calculate_cost(
                    model_name=model,
                    input_tokens=result.total_prompt_tokens,
                    output_tokens=result.total_completion_tokens
                )

            result_json: ResultJson = {
                "task_id": sample["task_id"],
                "model_answer": structured_output.model_answer if structured_output else None,
                "reasoning_trace": structured_output.reasoning_trace if structured_output else None,
            }

            # Calculate score
            score = question_scorer(
                model_answer=structured_output.model_answer if structured_output else "",
                ground_truth=sample.get("Final answer", "")
            )

            log: EvalLog = {
                "task_id": sample["task_id"],
                "file_name": sample.get("file_name"),
                "file_path": sample.get("file_path"),
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "level": sample.get("Level"),
                "final_answer": sample.get("Final answer", ""),
                "annotator_metadata": sample.get("Annotator Metadata"),
                "result": result_json,
                "score": score,
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

    def _serialize_metadata(self, state: GraphState) -> AgentMetadata:
        """Convert GraphState metadata to serializable AgentMetadata dict."""
        if state is None:
            return None  # type: ignore

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
        logs_list: List[EvalLog],
        logs_list_formatted: List[ResultJson]
    ) -> None:
        """
        Save evaluation logs to JSON files.

        Args:
            logs_list: Full logs with metadata (typed as EvalLog)
            logs_list_formatted: Formatted results for submission (typed as ResultJson)
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

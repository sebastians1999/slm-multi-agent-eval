from datasets import Dataset
from typing import Optional, List, Dict, Any, Union
from datetime import datetime
import os
from pathlib import Path
import json
from tqdm import tqdm
import re
import time

# Import multi-agent
from scr.agents.agent_01.multiAgent import MultiAgent
from scr.agents.agent_02.singleAgent import SingleAgent

# Import cost tracker
from scr.utilities.cost_tracker import CostTracker


class Eval_pipeline:
    """
    Evaluation pipeline for GSM8K (grade-school math) using a multi-agent solver.
    Computes accuracy by matching the model's final numeric answer to the gold label.
    """

    def __init__(
        self,
        dataset: Dataset,
        model: str,
        temperature: float = 0.3,
        base_url: Optional[str] = None,
        api_key: str = "",
        max_iterations: int = 2,
        use_web_search: bool = False,
        log_folder_path: Optional[str] = None,
        agent: Optional[Union[MultiAgent, SingleAgent]] = None,
    ):
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

        self.agent = agent or MultiAgent(
            model=model,
            temperature=temperature,
            base_url=base_url,
            api_key=api_key,
            max_iterations=max_iterations,
            use_web_search=use_web_search,
        )

        # Initialize cost tracker
        self.cost_tracker = CostTracker()

    # ---------------------------------------------------------------------

    def run_eval(self):
        """
        Run evaluation on GSM8K: iterate samples, call agent, extract final numeric
        answer, compare to gold, save logs, and print accuracy.
        """
        logs: List[Dict[str, Any]] = []
        logs_formatted: List[Dict[str, Any]] = []

        total = 0
        correct = 0

        for idx, sample in enumerate(tqdm(self.dataset, desc="Evaluating GSM8K")):
            # GSM8K fields
            question = sample.get("question")
            gold_answer_raw = sample.get("answer")  # usually "... #### 123"
            sample_id = sample.get("id", idx)

            if not question or gold_answer_raw is None:
                print(f"Skipping sample {sample_id} - missing question/answer")
                continue

            total += 1
            start_time = time.time()

            # Call agent
            result = self.call_agent(question=question)
            runtime_seconds = time.time() - start_time


            model_answer_text = result.get("solution")
            print("gold answer raw:", gold_answer_raw)
            print("model answer text:", model_answer_text)

            # Extract numeric answers
            # REPLACE IT WITH LLM EXTRACTING THE NUMBER

            pred_num = self._extract_gsm8k_number(model_answer_text)
            gold_num = self._extract_gsm8k_number(gold_answer_raw)

            print("gold number:", gold_num)
            print("extracted predicted number:", pred_num)
            is_correct = (pred_num is not None) and (gold_num is not None) and (pred_num == gold_num)
            if is_correct:
                correct += 1
                print("Correct! VAMOOOOOOSSS")

            metadata = result.get("metadata")
            cost_data = None
            if metadata and getattr(metadata, "prompt_tokens", None) and getattr(metadata, "completion_tokens", None):
                cost_data = self.cost_tracker.calculate_cost(
                    model_name=self.model,
                    input_tokens=metadata.prompt_tokens,
                    output_tokens=metadata.completion_tokens
                )

            # Minimal result JSON (for quick review / leaderboard-like)
            result_json = {
                "id": sample_id,
                "question": question,
                "gold_answer": gold_answer_raw,
                "gold_number": gold_num,
                "model_answer_text": model_answer_text,
                "pred_number": pred_num,
                "is_correct": is_correct,
                "runtime_seconds": runtime_seconds,
            }

            # Full log with metadata
            log = {
                "id": sample_id,
                "timestamp": datetime.now().isoformat(),
                "question": question,
                "gold_answer": gold_answer_raw,
                "prediction": {
                    "text": model_answer_text,
                    "number": pred_num,
                    "is_correct": is_correct,
                },
                "agent_metadata": self._serialize_metadata(metadata) if metadata else None,
                "cost_data": cost_data,
                "model_temperature": self.temperature,
                "model": self.model,
                "runtime_seconds": runtime_seconds,
            }

            logs.append(log)
            logs_formatted.append(result_json)

        # Save logs
        accuracy = (correct / total) if total > 0 else 0.0
        self.save_logs(logs_list=logs, logs_list_formatted=logs_formatted)

        print(f"\n✅ GSM8K Accuracy on {total} samples: {accuracy:.2%}")
        return {"accuracy": accuracy, "total": total, "correct": correct, "logs_path": self.log_folder_path}

    # ---------------------------------------------------------------------

    def call_agent(self, question: str) -> Dict[str, Any]:
        """Call the agent to solve a single question."""
        try:
            return self.agent.run(problem=question)
        except Exception as e:
            print(f"Error running agent on question: {e}")
            return {"structured_output": None, "metadata": None, "error": str(e)}

    # ---------------------------------------------------------------------


    def _extract_gsm8k_number(self, text: Optional[str]) -> Optional[str]:
        """
        Extract a clean numeric answer for GSM8K.
        - Prefer the '#### <number>' format.
        - Otherwise extract the last valid number.
        - Always return ONLY the digits (and optional minus sign or decimal point).
        No $, no commas, no parentheses.
        """
        if not text:
            return None

        text = str(text)

        # 1. First try GSM8K format: '#### <number>'
        m = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
        if m:
            raw = m.group(1)
            clean = re.sub(r"[^\d\.\-+]", "", raw)  # strip $, commas, parentheses, etc.
            return clean

        # 2. Otherwise find ALL numeric occurrences
        #    Allow optional $ and thousand separators, but strip them later.
        matches = re.findall(r"\$?[-+]?\d[\d,]*(?:\.\d+)?", text)
        if not matches:
            return None

        raw = matches[-1]  # last number
        clean = re.sub(r"[^\d\.\-+]", "", raw)  # remove $, commas, any non-numeric symbols

        return clean if clean else None

    def _serialize_metadata(self, metadata) -> Dict[str, Any]:
        """Convert AgentMetaData to serializable dict."""
        if metadata is None:
            return None
        return {
            "session_id": getattr(metadata, "session_id", None),
            "start_time": metadata.start_time.isoformat() if getattr(metadata, "start_time", None) else None,
            "end_time": metadata.end_time.isoformat() if getattr(metadata, "end_time", None) else None,
            "model_name": getattr(metadata, "model_name", None),
            "temperature": getattr(metadata, "temperature", None),
            "iterations": getattr(metadata, "iterations", None),
            "completion_tokens": getattr(metadata, "completion_tokens", None),
            "prompt_tokens": getattr(metadata, "prompt_tokens", None),
            "total_tokens": getattr(metadata, "total_tokens", None),
            "total_energy_joules": getattr(metadata, "total_energy_joules", None),
            "total_duration_seconds": getattr(metadata, "total_duration_seconds", None),
            "average_watts": getattr(metadata, "average_watts", None),
        }

    # ---------------------------------------------------------------------

    def save_logs(
        self,
        logs_list: List[Dict[str, Any]],
        logs_list_formatted: List[Dict[str, Any]]
    ) -> None:
        """
        Save evaluation logs to JSON files (full + compact).
        """
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = Path(self.log_folder_path) / f"gsm8k_eval_{timestamp}"
        os.makedirs(folder, exist_ok=True)

        file_path_logs = Path(folder) / "gsm8k_full_logs.json"
        with open(file_path_logs, "w") as f:
            json.dump(logs_list, f, indent=2)

        file_path_logs_formatted = Path(folder) / "gsm8k_results.json"
        with open(file_path_logs_formatted, "w") as f:
            json.dump(logs_list_formatted, f, indent=2)

        print(f"\nLogs saved to: {folder}")

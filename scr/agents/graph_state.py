from typing import Optional
from pydantic import BaseModel
from datetime import datetime
from .base_agent import agent_meta_data

class StructuredOutput(BaseModel):
   
    model_answer: Optional[str] 
    reasoning_trace: Optional[str] 


class GraphState(BaseModel):
    session_id: str
    problem: str
    structured_output: StructuredOutput
    start_time: datetime
    end_time: datetime
    model_name: str
    temperature: float
    iterations: int = 0
    total_completion_tokens: int = 0
    total_prompt_tokens: int = 0
    total_tokens: int = 0
    total_energy_joules: float = 0.0
    total_duration_seconds: float = 0.0
    average_watts: float = 0.0

    def accumulate_agent_metadata(self, metadata: agent_meta_data) -> None:
        self.total_completion_tokens += metadata.completion_tokens
        self.total_prompt_tokens += metadata.prompt_tokens
        self.total_tokens += metadata.total_tokens
        self.total_energy_joules += metadata.total_energy_joules
        self.total_duration_seconds += metadata.total_duration_seconds

        if self.total_duration_seconds > 0:
            self.average_watts = self.total_energy_joules / self.total_duration_seconds
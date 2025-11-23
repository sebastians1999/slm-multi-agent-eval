from dataclasses import dataclass
from typing import Optional, List


@dataclass
class AgentConfig:
    """Configuration for a single discussion agent"""
    agent_id: str
    model_name: str
    system_prompt: str
    tool_categories: Optional[List[str]] = None
    temperature: float = 0.8
    max_iterations: int = 10



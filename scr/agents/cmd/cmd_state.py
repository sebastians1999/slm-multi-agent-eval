from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime

from scr.agents.base_state import GraphState


class AgentResponse(BaseModel):
    """Response from a single agent in a discussion round"""
    agent_id: str
    viewpoint: str
    explanation: str
    round_number: int
    timestamp: datetime = Field(default_factory=datetime.now)
    tools_used: List[str] = Field(default_factory=list)


class DiscussionState(GraphState):
    """
    State of the CMD discussion.

    Extends GraphState for compatibility with eval_pipeline.
    Inherits all energy/token tracking fields and accumulate_agent_metadata() method.
    """

    # Inherited from GraphState (no need to redeclare):
    # - session_id: str
    # - problem: str  (maps to the discussion question)
    # - structured_output: StructuredOutput  (final answer goes here)
    # - start_time, end_time: datetime
    # - model_name: str
    # - temperature: float
    # - iterations: int
    # - total_completion_tokens, total_prompt_tokens, total_tokens: int
    # - total_energy_joules, total_duration_seconds, average_watts: float
    # - accumulate_agent_metadata(metadata) method

    # CMD-specific fields
    num_agents: int
    max_rounds: int
    current_round: int = 0
    active_agents: List[str]

    # Discussion history and results
    discussion_history: List[AgentResponse] = Field(default_factory=list)
    votes: Dict[str, str] = Field(default_factory=dict)  # agent_id -> viewpoint
    final_decision: Optional[str] = None
    final_explanation: Optional[str] = None
    is_tie: bool = False

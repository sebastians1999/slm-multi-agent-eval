"""
CMD (Conquer-and-Merge Discussion) Multi-Agent Framework

This module implements a simplified single-group CMD architecture where
multiple agents discuss a problem, exchange opinions, and reach a consensus.
"""

from .agent_config import AgentConfig
from .agent_group import AgentGroup
from .discussion_agent import DiscussionAgent
from .cmd_state import DiscussionState, AgentResponse
from .cmd_orchestrator import CMDOrchestrator
from .voting import count_votes, get_tied_viewpoints, get_vote_distribution

__all__ = [
    "AgentConfig",
    "AgentGroup",
    "DiscussionAgent",
    "DiscussionState",
    "AgentResponse",
    "CMDOrchestrator",
    "count_votes",
    "get_tied_viewpoints",
    "get_vote_distribution",
]

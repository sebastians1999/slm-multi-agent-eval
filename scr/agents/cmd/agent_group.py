from typing import List

from .agent_config import AgentConfig
from .discussion_agent import DiscussionAgent


class AgentGroup:
    """Manages a collection of discussion agents"""

    def __init__(self, configs: List[AgentConfig]):
        """Initialize agent group from configs"""
        self.configs = configs
        self.agents = []

        for config in configs:
            agent = DiscussionAgent(config)
            self.agents.append(agent)

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, idx):
        return self.agents[idx]

    def __iter__(self):
        return iter(self.agents)

    @property
    def agent_ids(self) -> List[str]:
        """Get list of all agent IDs"""
        return [agent.agent_id for agent in self.agents]

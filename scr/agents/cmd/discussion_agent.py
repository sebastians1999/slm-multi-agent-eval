from typing import Optional, List
import re
from datetime import datetime

from scr.agents.base_agent import BaseAgent
from scr.agents.graph_state import GraphState
from .agent_config import AgentConfig
from .cmd_state import AgentResponse


class DiscussionAgent(BaseAgent):
    """Discussion agent that extends BaseAgent for CMD"""

    def __init__(self, config: AgentConfig):
        # Initialize parent BaseAgent
        super().__init__(
            model=config.model_name,
            tool_categories=config.tool_categories,
            temperature=config.temperature,
            max_iterations=config.max_iterations
        )

        # CMD-specific attributes
        self.agent_id = config.agent_id
        self.system_prompt = config.system_prompt
        self.config = config

    def generate_response(
        self,
        question: str,
        previous_answer: Optional[AgentResponse],
        others_opinions: List[AgentResponse],
        round_number: int,
        use_tools: bool = True
    ) -> AgentResponse:
        """Generate response in discussion context"""

        # Build discussion prompt with system prompt
        prompt = self._build_discussion_prompt(
            question, previous_answer, others_opinions
        )

        # Prepend system prompt
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # Call parent invoke (with tool control)
        result = self.invoke(messages, tools=use_tools)

        # Parse structured output
        parsed = self._parse_response(result["content"])

        # Extract tool calls from result
        tools_used = []
        if result.get("tool_calls"):
            tools_used = [tc.get("name", "") for tc in result["tool_calls"]]

        return AgentResponse(
            agent_id=self.agent_id,
            viewpoint=parsed["viewpoint"],
            explanation=parsed["explanation"],
            round_number=round_number,
            timestamp=datetime.now(),
            tools_used=tools_used
        )

    def _build_discussion_prompt(
        self,
        question: str,
        previous_answer: Optional[AgentResponse],
        others_opinions: List[AgentResponse]
    ) -> str:
        """Build the discussion prompt with context"""

        parts = [f"# Main Question\n{question}\n"]

        # Add previous answer if not first round
        if previous_answer:
            parts.append(f"\n# Your Previous Answer (Round {previous_answer.round_number})")
            parts.append(f"Viewpoint: {previous_answer.viewpoint}")
            parts.append(f"Explanation: {previous_answer.explanation}\n")

        # Add others' opinions (full viewpoint + explanation for single group)
        if others_opinions:
            parts.append("\n# Other Agents' Opinions")
            for opinion in others_opinions:
                parts.append(f"\n## Agent {opinion.agent_id}:")
                parts.append(f"Viewpoint: {opinion.viewpoint}")
                parts.append(f"Explanation: {opinion.explanation}")

        # Instruction
        parts.append("\n# Your Task")
        parts.append(
            "Consider the question, your previous answer (if any), "
            "and other agents' opinions. Provide your updated answer in this format:\n\n"
            "VIEWPOINT: [Your concise answer to the question]\n"
            "EXPLANATION: [Your detailed reasoning and analysis]"
        )

        return "\n".join(parts)

    def run(self, prompt: str) -> GraphState:
        """
        Implement abstract run method (required by BaseAgent).

        Not used in CMD orchestrator, but required for BaseAgent inheritance.
        """
        raise NotImplementedError(
            "DiscussionAgent.run() is not used in CMD. "
            "Use CMDOrchestrator.run() instead."
        )

    def generate_vote(
        self,
        question: str,
        previous_answer: AgentResponse,
        others_final_opinions: List[AgentResponse]
    ) -> str:
        """
        Generate final vote with FINAL ANSWER format.

        Args:
            question: The original question
            previous_answer: This agent's final viewpoint from discussion
            others_final_opinions: Other agents' final viewpoints

        Returns:
            Full response content with FINAL ANSWER
        """
        prompt = self._build_voting_prompt(question, previous_answer, others_final_opinions)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        # No tools for voting
        result = self.invoke(messages, tools=False)

        return result["content"]

    def _build_voting_prompt(
        self,
        question: str,
        previous_answer: AgentResponse,
        others_final_opinions: List[AgentResponse]
    ) -> str:
        """
        Build voting prompt where agent must provide FINAL ANSWER.

        Different from discussion prompt:
        - Emphasizes final decision
        - Requires FINAL ANSWER format
        - Reviews all final viewpoints from discussion
        """
        parts = [
            f"# Question\n{question}\n",
            f"\n# Your Final Viewpoint from Discussion",
            f"Viewpoint: {previous_answer.viewpoint}",
            f"Explanation: {previous_answer.explanation}\n",
            f"\n# Other Agents' Final Viewpoints"
        ]

        for opinion in others_final_opinions:
            parts.append(f"\n## Agent {opinion.agent_id}:")
            parts.append(f"Viewpoint: {opinion.viewpoint}")

        parts.append("\n\n# Your Task")
        parts.append(
            "Based on the discussion above, provide your FINAL ANSWER to the question. "
            "Consider all viewpoints including your own. "
            "You MUST respond in this format:\n\n"
            "FINAL ANSWER: [your answer]\n\n"
            "YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. "
            "If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. "
            "If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise."
        )

        return "\n".join(parts)

    def _parse_response(self, content: str) -> dict:
        """Parse response into viewpoint and explanation"""

        # Try to extract VIEWPOINT and EXPLANATION
        viewpoint_match = re.search(r'VIEWPOINT:\s*(.+?)(?=\nEXPLANATION:|$)', content, re.DOTALL | re.IGNORECASE)
        explanation_match = re.search(r'EXPLANATION:\s*(.+)', content, re.DOTALL | re.IGNORECASE)

        if viewpoint_match and explanation_match:
            return {
                "viewpoint": viewpoint_match.group(1).strip(),
                "explanation": explanation_match.group(1).strip()
            }

        # Fallback: split on newlines or use entire content
        lines = content.strip().split('\n', 1)
        return {
            "viewpoint": lines[0].strip() if lines else content.strip(),
            "explanation": lines[1].strip() if len(lines) > 1 else content.strip()
        }

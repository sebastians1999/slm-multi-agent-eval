from typing import List, Optional
import uuid
import os
from datetime import datetime
from collections import Counter
import yaml
from .agent_group import AgentGroup
from .agent_config import AgentConfig
from .cmd_state import DiscussionState, AgentResponse
from .discussion_agent import DiscussionAgent
from scr.agents.base_state import StructuredOutput
from scr.agents.utils import extract_final_answer
import dotenv
from dotenv import load_dotenv


class CMDOrchestrator:
    """Orchestrates the CMD discussion workflow"""

    def __init__(
        self,
        agent_group: AgentGroup,
        max_rounds: int = 2,
        enable_secretary: bool = True,
        secretary_config: Optional[any] = None
    ):
        """
        Initialize CMD orchestrator

        Args:
            agent_group: Group of discussion agents
            max_rounds: Maximum number of discussion rounds
            enable_secretary: Whether to use secretary for tie-breaking
            secretary_config: Optional configuration for secretary agent
        """
        self.agent_group = agent_group
        self.max_rounds = max_rounds
        self.enable_secretary = enable_secretary
        self.secretary_config = secretary_config

    def run_discussion(self, question: str) -> DiscussionState:
        """
        Execute full CMD workflow

        Args:
            question: The question/problem to discuss

        Returns:
            Final discussion state with decision
        """
        # Initialize state with GraphState fields for eval_pipeline compatibility
        state = DiscussionState(
            # GraphState required fields
            session_id=str(uuid.uuid4()),
            problem=question,
            structured_output=StructuredOutput(
                model_answer=None,
                reasoning_trace=None
            ),
            start_time=datetime.now(),
            end_time=datetime.now(),
            model_name=self.agent_group[0].config.model_name,  # Use first agent's model for logging
            temperature=self.agent_group[0].config.temperature,  # Use first agent's temperature
            num_agents=len(self.agent_group),
            max_rounds=self.max_rounds,
            active_agents=self.agent_group.agent_ids
        )

        # Stage 1: Initial responses (round 0)
        print(f"\n=== CMD Discussion: Round 0 (Initial) ===")
        state = self._initialize_discussion(state)

        # Stage 2: Discussion rounds (1 to max_rounds)
        for round_num in range(1, self.max_rounds + 1):
            print(f"\n=== CMD Discussion: Round {round_num} ===")
            state.current_round = round_num
            state = self._run_discussion_round(state)

        # Stage 3: Voting round + Final decision
        print(f"\n=== CMD Discussion: Voting Round ===")
        state = self._run_voting_round(state)

        print(f"\n=== CMD Discussion: Determining Final Decision ===")
        state = self._determine_final_decision(state)

        # If tie, secretary resolves
        if state.is_tie and self.enable_secretary:
            print(f"=== Tie detected, secretary resolving ===")
            state = self._resolve_with_secretary(state)

        # Accumulate all agents' metadata (done once at the end to avoid double-counting)
        for agent in self.agent_group.agents:
            state.accumulate_agent_metadata(agent.meta_data)

        # Count total LLM invocations: discussion rounds + voting + secretary (if applicable)
        state.iterations = len(state.discussion_history) + len(state.votes)
        if state.is_tie and self.enable_secretary:
            state.iterations += 1  # Secretary invocation

        # Populate structured_output for eval_pipeline compatibility
        state.structured_output = StructuredOutput(
            model_answer=state.final_decision,
            reasoning_trace=self._build_discussion_summary(state)
        )

        # Update end time
        state.end_time = datetime.now()

        return state

    def run(self, problem: str) -> DiscussionState:
        """
        Run method compatible with eval_pipeline.

        This method signature matches what eval_pipeline expects:
        agent.run(problem=question) -> GraphState

        Args:
            problem: Question/problem to discuss

        Returns:
            DiscussionState (which extends GraphState)
        """
        return self.run_discussion(problem)

    def _initialize_discussion(self, state: DiscussionState) -> DiscussionState:
        """
        Stage 1: All agents generate initial responses (sequential for accurate energy tracking)

        Args:
            state: Current discussion state

        Returns:
            Updated state with initial responses
        """

        responses = []

        # Sequential execution (for accurate energy measurement)
        for agent in self.agent_group.agents:
            print(f"  Agent {agent.agent_id}: Generating initial response...")
            response = agent.generate_response(
                question=state.problem,  
                previous_answer=None,
                others_opinions=[],
                round_number=0
            )
            responses.append(response)

        state.discussion_history.extend(responses)
        return state

    def _run_discussion_round(self, state: DiscussionState) -> DiscussionState:
        """
        Stage 2: Run one discussion round with all agents (sequential for accurate energy tracking)

        Args:
            state: Current discussion state

        Returns:
            Updated state with round responses
        """

        responses = []

        # Sequential execution (for accurate energy measurement)
        for agent in self.agent_group.agents:
            # Get this agent's previous response
            prev = self._get_latest_response(state, agent.agent_id)

            # Get all OTHER agents' responses from previous round
            others = self._get_others_responses(
                state,
                agent.agent_id,
                state.current_round - 1
            )

            print(f"  Agent {agent.agent_id}: Considering {len(others)} other opinions...")
            response = agent.generate_response(
                question=state.problem,  # Use 'problem' from GraphState
                previous_answer=prev,
                others_opinions=others,
                round_number=state.current_round
            )
            responses.append(response)

        state.discussion_history.extend(responses)
        return state

    def _run_voting_round(self, state: DiscussionState) -> DiscussionState:
        """
        Stage 3a: Voting round where all agents provide FINAL ANSWER.

        Sequential execution for accurate energy tracking.

        Args:
            state: Current discussion state

        Returns:
            Updated state with votes
        """
        for agent in self.agent_group.agents:
            # Get agent's final viewpoint from last discussion round
            final_response = self._get_latest_response(state, agent.agent_id)

            if not final_response:
                print(f"  Warning: No final response found for {agent.agent_id}")
                continue


            print(f"  Agent {agent.agent_id}: Casting vote...")

            vote_content = agent.generate_vote(
                question=state.problem,
                previous_response=final_response,
            )

            # Debug: print full vote content
            print(f"\n    DEBUG - Full vote content from {agent.agent_id}:")
            print(f"    {vote_content[:500]}")  # First 500 chars
            print()

            # Extract FINAL ANSWER
            extracted_vote = extract_final_answer(vote_content)

            if extracted_vote:
                state.votes[agent.agent_id] = extracted_vote
                print(f"    Vote: {extracted_vote}")
            else:
                print(f"    Warning: No FINAL ANSWER found in vote")
                # Fallback to first 100 chars of content
                state.votes[agent.agent_id] = vote_content.strip()[:100]

        return state

    def _determine_final_decision(self, state: DiscussionState) -> DiscussionState:
        """
        Stage 3b: Count votes and determine if there's a majority.

        Args:
            state: Current discussion state

        Returns:
            Updated state with final decision or tie status
        """
        vote_counts = Counter(state.votes.values())

        if not vote_counts:
            state.is_tie = True
            print("  Warning: No votes collected")
            return state

        max_count = max(vote_counts.values())
        winners = [v for v, c in vote_counts.items() if c == max_count]

        if len(winners) == 1:
            # Majority reached
            state.final_decision = winners[0]
            state.is_tie = False
            print(f"  Majority decision: {state.final_decision} ({max_count}/{len(state.active_agents)} votes)")
        else:
            # Tie - secretary needed
            state.is_tie = True
            print(f"  Tie between: {', '.join(winners)}")

        return state

    def _resolve_with_secretary(self, state: DiscussionState) -> DiscussionState:
        """
        Resolve tie using secretary agent

        Args:
            state: Current discussion state with tie

        Returns:
            Updated state with secretary's decision
        """
        # Load secretary prompt from YAML
        prompt_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "prompts",
            "prompts.yaml"
        )
        with open(prompt_path, "r") as f:
            prompts = yaml.safe_load(f)
        secretary_prompt = prompts["secretary_prompt"]

        load_dotenv()
        base_url = os.getenv("MODAL_BASE_URL")

        # Create secretary agent
        secretary_config = self.secretary_config or AgentConfig(
            agent_id="secretary",
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            system_prompt=secretary_prompt,
            tool_categories=[],  # No tools for secretary
            temperature=0.5,
            base_url=base_url,
            api_key="EMPTY"

        )

        secretary = DiscussionAgent(secretary_config)

        # Get all final responses for secretary to review
        final_responses = [
            self._get_latest_response(state, agent_id)
            for agent_id in state.active_agents
        ]

        # Secretary reviews all arguments (without tools)
        print("  Secretary reviewing all arguments...")
        secretary_response = secretary.generate_response(
            question=state.problem,
            previous_answer=None,
            others_opinions=[r for r in final_responses if r is not None],
            round_number=state.current_round + 1,
            use_tools=False  # Secretary doesn't use tools
        )

        # Extract FINAL ANSWER from secretary's response
        # Get full content from viewpoint + explanation
        full_secretary_content = f"{secretary_response.viewpoint}\n{secretary_response.explanation}"
        extracted_answer = extract_final_answer(full_secretary_content)

        # Use extracted answer if found, otherwise fall back to viewpoint
        state.final_decision = extracted_answer if extracted_answer else secretary_response.viewpoint
        state.final_explanation = secretary_response.explanation
        state.is_tie = False

        print(f"  Secretary decision: {state.final_decision}")

        return state

    def _get_latest_response(
        self,
        state: DiscussionState,
        agent_id: str
    ) -> Optional[AgentResponse]:
        """Get the most recent response from a specific agent"""
        agent_responses = [
            r for r in state.discussion_history
            if r.agent_id == agent_id
        ]
        return agent_responses[-1] if agent_responses[-1] else None

    def _get_others_responses(
        self,
        state: DiscussionState,
        agent_id: str,
        round_number: int
    ) -> List[AgentResponse]:
        """Get all other agents' responses from a specific round"""
        return [
            r for r in state.discussion_history
            if r.agent_id != agent_id and r.round_number == round_number
        ]

    def _build_discussion_summary(self, state: DiscussionState) -> str:
        """
        Build a summary of the discussion for reasoning_trace.

        Args:
            state: Current discussion state

        Returns:
            String summary of the discussion
        """
        summary_parts = [
            f"CMD Discussion Summary",
            f"Question: {state.problem}",
            f"Agents: {state.num_agents}",
            f"Rounds: {state.current_round}",
            f"\nFinal Decision: {state.final_decision}",
            f"\nVote Distribution:"
        ]
        # Add vote counts
        vote_counts = Counter(state.votes.values())
        for viewpoint, count in vote_counts.items():
            summary_parts.append(f"  - {viewpoint}: {count} votes")

        # Add brief discussion history
        summary_parts.append(f"\nDiscussion History:")
        for response in state.discussion_history[-state.num_agents:]:  # Last round only
            summary_parts.append(f"  [{response.agent_id}] {response.viewpoint}")

        return "\n".join(summary_parts)

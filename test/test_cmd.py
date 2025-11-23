import pytest
from scr.agents.cmd import (
    AgentConfig,
    AgentGroup,
    CMDOrchestrator,
    DiscussionState
)


def test_agent_config_creation():
    """Test creating agent configurations"""
    config = AgentConfig(
        agent_id="test_agent",
        model_name="Qwen/Qwen3-4B-Instruct-2507",
        system_prompt="You are a test agent.",
        tool_categories=["search"],
        temperature=0.7
    )

    assert config.agent_id == "test_agent"
    assert config.model_name == "Qwen/Qwen3-4B-Instruct-2507"
    assert config.tool_categories == ["search"]


def test_agent_group_creation():
    """Test creating agent group from configs"""
    configs = [
        AgentConfig(
            agent_id="agent_1",
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            system_prompt="You are agent 1.",
            tool_categories=[]  # No tools needed for this test
        ),
        AgentConfig(
            agent_id="agent_2",
            model_name="microsoft/Phi-3.5-mini-instruct",
            system_prompt="You are agent 2.",
            tool_categories=[]  # No tools needed for this test
        ),
    ]

    group = AgentGroup(configs)

    assert len(group) == 2
    assert group.agent_ids == ["agent_1", "agent_2"]


def test_discussion_state_initialization():
    """Test discussion state initialization"""
    state = DiscussionState(
        question="What is 2+2?",
        num_agents=3,
        active_agents=["agent_1", "agent_2", "agent_3"],
        max_rounds=2
    )

    assert state.question == "What is 2+2?"
    assert state.num_agents == 3
    assert state.current_round == 0
    assert len(state.discussion_history) == 0


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Integration test - requires API access. Set RUN_INTEGRATION_TESTS=1 to run."
)
def test_cmd_discussion_integration():
    """Integration test - run full CMD discussion"""
    import os

    # Create simple agent group
    configs = [
        AgentConfig(
            agent_id="agent_1",
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            system_prompt="You are a helpful assistant. Be concise.",
            tool_categories=[],  # No tools needed for simple discussion test
            temperature=0.7
        ),
        AgentConfig(
            agent_id="agent_2",
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            system_prompt="You are a critical thinker. Question assumptions.",
            tool_categories=[],  # No tools needed for simple discussion test
            temperature=0.7
        ),
        AgentConfig(
            agent_id="agent_3",
            model_name="Qwen/Qwen3-4B-Instruct-2507",
            system_prompt="You are an optimist. Find positive solutions.",
            tool_categories=[],  # No tools needed for simple discussion test
            temperature=0.7
        ),
    ]

    group = AgentGroup(configs)
    orchestrator = CMDOrchestrator(group, max_rounds=1, enable_secretary=True)

    # Run discussion
    result = orchestrator.run_discussion("What is the capital of France?")

    # Verify results
    assert result.final_decision is not None
    assert len(result.discussion_history) >= 3  # At least initial responses
    assert len(result.votes) == 3  # All agents voted
    print(f"\nFinal decision: {result.final_decision}")
    print(f"Total responses: {len(result.discussion_history)}")


# Import os for skipif
import os

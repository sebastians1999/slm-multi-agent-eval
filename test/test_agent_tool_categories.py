"""
Integration Tests for BaseAgent with Dynamic Tool Loading

Tests agent initialization with different tool category configurations.
"""

import pytest
from scr.agents.base_agent import BaseAgent


# Mock agent implementation for testing
class TestAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing."""

    def run(self, prompt):
        """Mock run method."""
        return {"result": "test"}


def test_agent_with_all_tools():
    """Test agent initialization with all tools (default)."""
    agent = TestAgent(model="gpt-4", api_key="test")

    # Should have tools loaded
    assert agent.tools is not None
    assert agent.tool_functions is not None
    assert len(agent.tools) > 0
    assert len(agent.tool_functions) > 0

    # Should have browser manager if browser tools available
    # (may be None if playwright not installed)
    agent.cleanup()


def test_agent_with_search_category_only():
    """Test agent with only search tools."""
    agent = TestAgent(
        model="gpt-4",
        api_key="test",
        tool_categories=['search']
    )

    # Should have search tools
    assert len(agent.tools) > 0
    tool_names = [t['function']['name'] for t in agent.tools]
    assert 'tavily_search' in tool_names

    # Should not have browser manager
    assert agent.browser_manager is None

    agent.cleanup()


def test_agent_with_code_category_only():
    """Test agent with only code execution tools."""
    agent = TestAgent(
        model="gpt-4",
        api_key="test",
        tool_categories=['code']
    )

    # Should have code tools
    assert len(agent.tools) > 0
    tool_names = [t['function']['name'] for t in agent.tools]
    assert 'execute_python' in tool_names

    # Should not have browser manager
    assert agent.browser_manager is None

    agent.cleanup()


def test_agent_with_multiple_categories():
    """Test agent with multiple specific categories."""
    agent = TestAgent(
        model="gpt-4",
        api_key="test",
        tool_categories=['search', 'code']
    )

    # Should have both search and code tools
    assert len(agent.tools) >= 2
    tool_names = [t['function']['name'] for t in agent.tools]
    assert 'tavily_search' in tool_names
    assert 'execute_python' in tool_names

    # Should not have browser manager (browser not requested)
    assert agent.browser_manager is None

    agent.cleanup()


@pytest.mark.skipif(
    True,  # Skip by default as it requires playwright
    reason="Requires playwright installation"
)
def test_agent_with_browser_category():
    """Test agent with browser tools."""
    agent = TestAgent(
        model="gpt-4",
        api_key="test",
        tool_categories=['browser']
    )

    # Should have browser tools
    assert len(agent.tools) > 0
    tool_names = [t['function']['name'] for t in agent.tools]
    # Check for common PlayWright tools
    assert any('navigate' in name.lower() for name in tool_names)

    # Should have browser manager
    assert agent.browser_manager is not None

    agent.cleanup()


def test_agent_cleanup():
    """Test that agent cleanup works properly."""
    agent = TestAgent(model="gpt-4", api_key="test")

    # Should not raise errors
    agent.cleanup()
    # Multiple cleanups should be safe
    agent.cleanup()


def test_agent_context_manager():
    """Test using agent as context manager."""
    with TestAgent(model="gpt-4", api_key="test") as agent:
        assert agent is not None
        assert agent.tools is not None

    # Cleanup should have been called automatically


def test_agent_invalid_category():
    """Test that invalid category raises error."""
    with pytest.raises(ValueError, match="Unknown category"):
        TestAgent(
            model="gpt-4",
            api_key="test",
            tool_categories=['nonexistent']
        )


def test_agent_tool_functions_callable():
    """Test that all tool functions are callable."""
    agent = TestAgent(
        model="gpt-4",
        api_key="test",
        tool_categories=['search']
    )

    for tool_name, tool_func in agent.tool_functions.items():
        assert callable(tool_func), f"Tool {tool_name} is not callable"

    agent.cleanup()


def test_agent_tool_schemas_valid():
    """Test that all tool schemas have required fields."""
    agent = TestAgent(
        model="gpt-4",
        api_key="test",
        tool_categories=['search', 'code']
    )

    for tool in agent.tools:
        # Check OpenAI schema format
        assert tool['type'] == 'function'
        assert 'function' in tool

        func = tool['function']
        assert 'name' in func
        assert 'description' in func
        assert 'parameters' in func

        params = func['parameters']
        assert params['type'] == 'object'
        assert 'properties' in params
        assert 'required' in params

    agent.cleanup()


def test_agent_metadata_tracking():
    """Test that agent metadata is initialized correctly."""
    agent = TestAgent(model="gpt-4", api_key="test")

    # Check metadata initialization
    assert agent.meta_data is not None
    assert agent.meta_data.total_tokens == 0
    assert agent.meta_data.total_energy_joules == 0.0

    agent.cleanup()


def test_agent_preserves_existing_parameters():
    """Test that agent still accepts and preserves existing parameters."""
    agent = TestAgent(
        model="gpt-4",
        api_key="custom-key",
        temperature=0.7,
        base_url="https://custom.api",
        max_iterations=5,
        custom_param="value"
    )

    assert agent.model == "gpt-4"
    assert agent.api_key == "custom-key"
    assert agent.temperature == 0.7
    assert agent.base_url == "https://custom.api"
    assert agent.max_iterations == 5
    assert agent.config["custom_param"] == "value"

    agent.cleanup()


def test_empty_tool_categories():
    """Test behavior with empty tool categories list."""
    # Empty list should load no tools
    agent = TestAgent(
        model="gpt-4",
        api_key="test",
        tool_categories=[]
    )

    assert len(agent.tools) == 0
    assert len(agent.tool_functions) == 0

    agent.cleanup()

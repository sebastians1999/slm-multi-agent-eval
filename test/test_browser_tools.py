"""
Test suite for browser tools integration.

Tests the PlayWright browser toolkit integration with BaseAgent,
including navigation, text extraction, and cleanup.
"""

import os
import pytest
from scr.agents.base_agent import BaseAgent


class BrowserAgent(BaseAgent):
    """Simple agent implementation for testing browser tools."""

    def run(self, user_message: str):
        """Run the agent with a user message."""
        messages = [{"role": "user", "content": user_message}]
        return self.invoke(messages)


@pytest.fixture
def browser_agent():
    """
    Fixture that provides an agent with browser tools.

    Cleans up browser resources after test completes.
    """
    agent = BrowserAgent(
        model="gpt-4",  # Model doesn't matter for these tests
        api_key="test",
        tool_categories=['browser'],
        max_iterations=5
    )
    yield agent
    agent.cleanup()


def test_browser_tools_loaded(browser_agent):
    """Test that browser tools are properly loaded."""
    # Check that tools are loaded
    assert len(browser_agent.tools) > 0, "Browser tools should be loaded"

    # Check that browser-specific tools are present
    tool_names = [tool['function']['name'] for tool in browser_agent.tools]

    # Should have common browser tools
    expected_tools = ['navigate_browser', 'extract_text', 'current_webpage']
    for expected in expected_tools:
        assert any(expected in name for name in tool_names), \
            f"Should have a tool containing '{expected}'"

    print(f"\n✓ Loaded {len(browser_agent.tools)} browser tools")
    print(f"  Tool names: {tool_names}")


def test_browser_manager_exists(browser_agent):
    """Test that browser manager is properly initialized."""
    assert browser_agent.browser_manager is not None, \
        "Browser manager should be initialized"

    # Browser should be lazily initialized (not created yet)
    assert browser_agent.browser_manager._browser is None, \
        "Browser should not be created until first tool use"

    print("\n✓ Browser manager initialized (lazy loading)")


def test_browser_tool_functions_registered(browser_agent):
    """Test that all tool functions are properly registered."""
    # Check that all tool schemas have corresponding functions
    for tool in browser_agent.tools:
        tool_name = tool['function']['name']
        assert tool_name in browser_agent.tool_functions, \
            f"Tool '{tool_name}' should have a registered function"
        assert callable(browser_agent.tool_functions[tool_name]), \
            f"Tool function '{tool_name}' should be callable"

    print(f"\n✓ All {len(browser_agent.tools)} tools have callable functions")


def test_browser_tool_schemas_valid(browser_agent):
    """Test that all browser tool schemas are properly formatted."""
    for tool in browser_agent.tools:
        # Check OpenAI format
        assert tool['type'] == 'function', "Tool type should be 'function'"
        assert 'function' in tool, "Tool should have 'function' key"

        func = tool['function']
        assert 'name' in func, "Function should have 'name'"
        assert 'description' in func, "Function should have 'description'"
        assert 'parameters' in func, "Function should have 'parameters'"

        params = func['parameters']
        assert params['type'] == 'object', "Parameters type should be 'object'"
        assert 'properties' in params, "Parameters should have 'properties'"
        assert 'required' in params, "Parameters should have 'required' list"
        assert params.get('additionalProperties') is False, \
            "Should have additionalProperties: False"

    print(f"\n✓ All {len(browser_agent.tools)} tool schemas are valid")


@pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION_TESTS") != "1",
    reason="Requires running LLM server and takes time. Set RUN_INTEGRATION_TESTS=1 to enable"
)
def test_browser_navigation_integration():
    """
    Integration test: Agent navigates to a URL and extracts information.

    This test actually calls the LLM and browser tools, so it's skipped by default.
    Run with: RUN_INTEGRATION_TESTS=1 pytest test/test_browser_tools.py::test_browser_navigation_integration -v -s
    """
    agent = BrowserAgent(
        model="Qwen/Qwen3-4B-Instruct-2507",
        base_url="https://sebastian-schmuelling--slm-server-vllmserver-serve-dev.modal.run/v1/",
        temperature=0,
        tool_categories=['browser'],
        max_iterations=10
    )

    try:
        # Ask agent to navigate and extract info
        response = agent.run(
            "Navigate to https://example.com and tell me the main heading text."
        )

        # Verify tool was used
        assert response['iterations'] > 1, "Agent should have used browser tools"

        # Check content mentions example.com
        content_lower = response['content'].lower()
        assert 'example' in content_lower or 'domain' in content_lower, \
            "Response should contain info from example.com"

        # Check tool usage metadata
        assert agent.meta_data.tool_call_count > 0, "Should have tracked tool calls"
        assert len(agent.meta_data.unique_tools_used) > 0, \
            "Should have unique tools tracked"

        # Print results
        print(f"\n✓ Navigation test passed!")
        print(f"  Iterations: {response['iterations']}")
        print(f"  Tools used: {agent.meta_data.unique_tools_used}")
        print(f"  Total tool calls: {agent.meta_data.tool_call_count}")

    finally:
        agent.cleanup()


def test_browser_cleanup(browser_agent):
    """Test that browser cleanup works properly."""
    # Cleanup should work even if browser never initialized
    browser_agent.cleanup()

    # Should be safe to call multiple times
    browser_agent.cleanup()
    browser_agent.cleanup()

    print("\n✓ Browser cleanup works (even without initialization)")


def test_browser_context_manager():
    """Test using agent with browser tools as context manager."""
    with BrowserAgent(
        model="test",
        api_key="test",
        tool_categories=['browser']
    ) as agent:
        # Agent should be usable
        assert agent is not None
        assert len(agent.tools) > 0
        assert agent.browser_manager is not None

    # Browser should be cleaned up after context exit
    # (If it was ever initialized)
    print("\n✓ Context manager cleanup works")


def test_browser_tools_metadata_structure():
    """Test that browser tools have expected metadata structure."""
    agent = BrowserAgent(
        model="test",
        api_key="test",
        tool_categories=['browser']
    )

    # Check metadata is initialized
    assert agent.meta_data.tool_call_count == 0
    assert len(agent.meta_data.tool_calls) == 0
    assert len(agent.meta_data.unique_tools_used) == 0

    agent.cleanup()
    print("\n✓ Metadata structure is correct")


def test_only_browser_tools_loaded():
    """Test that only browser tools are loaded when specified."""
    agent = BrowserAgent(
        model="test",
        api_key="test",
        tool_categories=['browser']
    )

    tool_names = [tool['function']['name'] for tool in agent.tools]

    # Should NOT have search or code tools
    assert not any('search' in name.lower() for name in tool_names), \
        "Should not have search tools"
    assert not any('python' in name.lower() for name in tool_names), \
        "Should not have Python execution tools"

    # Should have browser-related tools
    assert any('browser' in name.lower() or 'navigate' in name.lower() or
               'webpage' in name.lower() for name in tool_names), \
        "Should have browser-related tools"

    agent.cleanup()
    print(f"\n✓ Only browser tools loaded (no search/code tools)")


def test_browser_tools_with_other_categories():
    """Test loading browser tools alongside other categories."""
    agent = BrowserAgent(
        model="test",
        api_key="test",
        tool_categories=['browser', 'search']
    )

    tool_names = [tool['function']['name'] for tool in agent.tools]

    # Should have both browser and search tools
    has_browser = any('browser' in name.lower() or 'navigate' in name.lower()
                     for name in tool_names)
    has_search = any('search' in name.lower() for name in tool_names)

    assert has_browser, "Should have browser tools"
    assert has_search, "Should have search tools"

    agent.cleanup()
    print(f"\n✓ Browser + search tools loaded together")
    print(f"  Total tools: {len(agent.tools)}")


if __name__ == "__main__":
    """Run tests with pytest."""
    print("\nTo run these tests:")
    print("  pytest test/test_browser_tools.py -v")
    print("\nTo run integration test:")
    print("  RUN_INTEGRATION_TESTS=1 pytest test/test_browser_tools.py::test_browser_navigation_integration -v -s")

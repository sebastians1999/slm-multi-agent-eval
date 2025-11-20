"""
Tests for Tool Registry System

Tests the dynamic tool loading and category management functionality.
"""

import pytest
from scr.agents.utils.tool_registry import ToolRegistry


@pytest.fixture
def clean_registry():
    """Fixture that provides a clean registry for each test."""
    registry = ToolRegistry()
    registry.clear()
    yield registry
    registry.clear()


def test_registry_singleton():
    """Test that registry follows singleton pattern."""
    registry1 = ToolRegistry()
    registry2 = ToolRegistry()
    assert registry1 is registry2


def test_register_category(clean_registry):
    """Test basic category registration."""
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    functions = {"test_tool": lambda: "result"}

    clean_registry.register_category("test", tools, functions)

    assert clean_registry.category_exists("test")
    assert "test" in clean_registry.list_categories()


def test_register_duplicate_category(clean_registry):
    """Test that registering duplicate category raises error."""
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "Test",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    functions = {"test_tool": lambda: "result"}

    clean_registry.register_category("test", tools, functions)

    with pytest.raises(ValueError, match="already registered"):
        clean_registry.register_category("test", tools, functions)


def test_register_mismatched_tools_functions(clean_registry):
    """Test that mismatched tools and functions raise error."""
    tools = [{
        "type": "function",
        "function": {
            "name": "tool1",
            "description": "Test",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    # Function name doesn't match tool name
    functions = {"tool2": lambda: "result"}

    with pytest.raises(ValueError, match="mismatch"):
        clean_registry.register_category("test", tools, functions)


def test_get_tools_all_categories(clean_registry):
    """Test getting tools from all categories."""
    # Register multiple categories
    tools1 = [{
        "type": "function",
        "function": {
            "name": "tool1",
            "description": "Tool 1",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    tools2 = [{
        "type": "function",
        "function": {
            "name": "tool2",
            "description": "Tool 2",
            "parameters": {"type": "object", "properties": {}}
        }
    }]

    clean_registry.register_category("cat1", tools1, {"tool1": lambda: "1"})
    clean_registry.register_category("cat2", tools2, {"tool2": lambda: "2"})

    # Get all tools (None parameter)
    all_tools, all_functions = clean_registry.get_tools_by_categories(None)

    assert len(all_tools) == 2
    assert len(all_functions) == 2
    assert "tool1" in all_functions
    assert "tool2" in all_functions


def test_get_tools_specific_categories(clean_registry):
    """Test getting tools from specific categories."""
    tools1 = [{
        "type": "function",
        "function": {
            "name": "tool1",
            "description": "Tool 1",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    tools2 = [{
        "type": "function",
        "function": {
            "name": "tool2",
            "description": "Tool 2",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    tools3 = [{
        "type": "function",
        "function": {
            "name": "tool3",
            "description": "Tool 3",
            "parameters": {"type": "object", "properties": {}}
        }
    }]

    clean_registry.register_category("cat1", tools1, {"tool1": lambda: "1"})
    clean_registry.register_category("cat2", tools2, {"tool2": lambda: "2"})
    clean_registry.register_category("cat3", tools3, {"tool3": lambda: "3"})

    # Get only cat1 and cat3
    selected_tools, selected_functions = clean_registry.get_tools_by_categories(["cat1", "cat3"])

    assert len(selected_tools) == 2
    assert len(selected_functions) == 2
    assert "tool1" in selected_functions
    assert "tool3" in selected_functions
    assert "tool2" not in selected_functions


def test_get_tools_unknown_category(clean_registry):
    """Test that requesting unknown category raises error."""
    clean_registry.register_category("cat1", [], {})

    with pytest.raises(ValueError, match="Unknown category"):
        clean_registry.get_tools_by_categories(["unknown"])


def test_category_metadata(clean_registry):
    """Test storing and retrieving category metadata."""
    tools = [{
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "Test",
            "parameters": {"type": "object", "properties": {}}
        }
    }]
    metadata = {"key": "value", "number": 42}

    clean_registry.register_category("test", tools, {"test_tool": lambda: "x"}, metadata)

    retrieved_metadata = clean_registry.get_category_metadata("test")
    assert retrieved_metadata == metadata


def test_get_all_metadata(clean_registry):
    """Test getting merged metadata from multiple categories."""
    clean_registry.register_category(
        "cat1", [], {},
        metadata={"key1": "value1"}
    )
    clean_registry.register_category(
        "cat2", [], {},
        metadata={"key2": "value2"}
    )

    all_metadata = clean_registry.get_all_metadata()
    assert all_metadata == {"key1": "value1", "key2": "value2"}


def test_list_categories(clean_registry):
    """Test listing all registered categories."""
    clean_registry.register_category("cat1", [], {})
    clean_registry.register_category("cat2", [], {})
    clean_registry.register_category("cat3", [], {})

    categories = clean_registry.list_categories()
    assert set(categories) == {"cat1", "cat2", "cat3"}


def test_category_exists(clean_registry):
    """Test checking if category exists."""
    clean_registry.register_category("existing", [], {})

    assert clean_registry.category_exists("existing")
    assert not clean_registry.category_exists("nonexistent")

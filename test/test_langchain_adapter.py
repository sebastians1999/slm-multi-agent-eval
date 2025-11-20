"""
Tests for LangChain Toolkit Adapter

Tests conversion of LangChain tools to OpenAI-compatible format.
"""

import pytest
from typing import Optional
from pydantic import BaseModel, Field
from scr.agents.utils.langchain_adapter import (
    convert_langchain_tools,
    _clean_pydantic_schema,
    _create_error_wrapped_function
)


# Mock LangChain tool for testing
class MockToolArgs(BaseModel):
    """Mock tool arguments schema."""
    query: str = Field(description="The query string")
    max_results: Optional[int] = Field(default=5, description="Max results")


class MockTool:
    """Mock LangChain BaseTool."""

    def __init__(self, name: str, description: str, has_args: bool = True):
        self.name = name
        self.description = description
        self.args_schema = MockToolArgs if has_args else None

    def _run(self, **kwargs):
        """Mock tool execution."""
        return f"Result for {kwargs}"


def test_clean_pydantic_schema():
    """Test cleaning Pydantic schema for OpenAI compatibility."""
    pydantic_schema = {
        "title": "ShouldBeRemoved",
        "type": "object",
        "properties": {
            "field1": {"type": "string", "title": "AlsoRemoved"},
            "field2": {"type": "integer"}
        },
        "required": ["field1"],
        "definitions": {"ShouldBeRemoved": {}}
    }

    cleaned = _clean_pydantic_schema(pydantic_schema)

    # Check removed fields
    assert "title" not in cleaned
    assert "definitions" not in cleaned

    # Check preserved fields
    assert cleaned["type"] == "object"
    assert "field1" in cleaned["properties"]
    assert "field2" in cleaned["properties"]
    assert cleaned["required"] == ["field1"]

    # Check added fields
    assert cleaned["additionalProperties"] is False

    # Check nested cleaning
    assert "title" not in cleaned["properties"]["field1"]


def test_clean_pydantic_schema_with_allof():
    """Test cleaning schema with allOf structures."""
    schema_with_allof = {
        "allOf": [
            {"type": "object", "properties": {"field1": {"type": "string"}}},
            {"required": ["field1"]}
        ]
    }

    cleaned = _clean_pydantic_schema(schema_with_allof)

    assert "allOf" not in cleaned
    assert cleaned["type"] == "object"
    assert "field1" in cleaned["properties"]


def test_create_error_wrapped_function():
    """Test error wrapping for tool functions."""

    def successful_func(x: int) -> int:
        return x * 2

    def failing_func(x: int) -> int:
        raise ValueError("Test error")

    # Test successful execution
    wrapped_success = _create_error_wrapped_function("test_tool", successful_func)
    result = wrapped_success(x=5)
    assert result == "10"

    # Test error handling
    wrapped_fail = _create_error_wrapped_function("test_tool", failing_func)
    result = wrapped_fail(x=5)
    assert "Error in tool 'test_tool'" in result
    assert "ValueError" in result
    assert "Test error" in result


def test_convert_langchain_tools_single_tool():
    """Test converting a single LangChain tool."""
    mock_tool = MockTool("search_tool", "Search the web")
    tools = [mock_tool]

    schemas, functions = convert_langchain_tools(tools)

    # Check schemas
    assert len(schemas) == 1
    schema = schemas[0]
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "search_tool"
    assert schema["function"]["description"] == "Search the web"
    assert "parameters" in schema["function"]

    # Check parameters are cleaned
    params = schema["function"]["parameters"]
    assert "title" not in params
    assert params["additionalProperties"] is False

    # Check functions
    assert len(functions) == 1
    assert "search_tool" in functions
    assert callable(functions["search_tool"])


def test_convert_langchain_tools_multiple_tools():
    """Test converting multiple LangChain tools."""
    tools = [
        MockTool("tool1", "First tool"),
        MockTool("tool2", "Second tool"),
        MockTool("tool3", "Third tool"),
    ]

    schemas, functions = convert_langchain_tools(tools)

    assert len(schemas) == 3
    assert len(functions) == 3
    assert "tool1" in functions
    assert "tool2" in functions
    assert "tool3" in functions


def test_convert_langchain_tools_no_args():
    """Test converting tool with no arguments."""
    mock_tool = MockTool("simple_tool", "Simple tool", has_args=False)
    tools = [mock_tool]

    schemas, functions = convert_langchain_tools(tools)

    assert len(schemas) == 1
    params = schemas[0]["function"]["parameters"]
    assert params["type"] == "object"
    assert params["properties"] == {}
    assert params["required"] == []


def test_converted_function_execution():
    """Test that converted functions can be executed."""
    mock_tool = MockTool("test_tool", "Test tool")
    tools = [mock_tool]

    _, functions = convert_langchain_tools(tools)

    # Execute the converted function
    result = functions["test_tool"](query="test", max_results=3)

    # Should return string (wrapped result)
    assert isinstance(result, str)
    assert "query" in result
    assert "test" in result


def test_converted_function_error_handling():
    """Test error handling in converted functions."""

    class FailingTool(MockTool):
        def _run(self, **kwargs):
            raise RuntimeError("Tool execution failed")

    failing_tool = FailingTool("failing_tool", "This tool fails")
    tools = [failing_tool]

    _, functions = convert_langchain_tools(tools)

    # Execute the function (should not raise, returns error string)
    result = functions["failing_tool"](query="test")

    assert isinstance(result, str)
    assert "Error in tool 'failing_tool'" in result
    assert "RuntimeError" in result


def test_convert_tool_missing_attributes():
    """Test converting tool with missing required attributes."""

    class BrokenTool:
        name = "broken"
        description = "Broken tool"
        # Missing args_schema and _run

    broken_tool = BrokenTool()
    tools = [broken_tool]

    # Should handle gracefully and skip broken tool
    schemas, functions = convert_langchain_tools(tools)

    # Broken tool should be skipped
    assert len(schemas) == 0
    assert len(functions) == 0


def test_schema_parameter_properties():
    """Test that all parameter properties are preserved."""

    class DetailedArgs(BaseModel):
        text: str = Field(description="Input text", min_length=1, max_length=100)
        count: int = Field(default=1, description="Count", ge=1, le=10)

    class DetailedTool(MockTool):
        def __init__(self):
            super().__init__("detailed", "Detailed tool")
            self.args_schema = DetailedArgs

    tool = DetailedTool()
    schemas, _ = convert_langchain_tools([tool])

    params = schemas[0]["function"]["parameters"]
    assert "text" in params["properties"]
    assert "count" in params["properties"]
    assert params["properties"]["text"]["description"] == "Input text"
    assert params["properties"]["count"]["description"] == "Count"

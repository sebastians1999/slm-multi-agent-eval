"""
LangChain Toolkit Adapter

Converts LangChain BaseTool instances to OpenAI-compatible tool schemas
and Python callables, enabling plug-and-play integration of any LangChain
toolkit into the agent framework.
"""

from typing import Dict, List, Callable, Tuple, Any, Optional
import copy


def _clean_pydantic_schema(pydantic_schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean Pydantic schema to ensure OpenAI compatibility.

    Pydantic schemas may contain fields incompatible with OpenAI function
    calling format. This function removes/transforms problematic fields.

    Args:
        pydantic_schema: Raw Pydantic model schema from .schema()

    Returns:
        Cleaned schema compatible with OpenAI function calling

    Transformations:
    - Removes 'title', 'definitions', 'allOf', '$ref' fields
    - Flattens 'allOf' structures
    - Ensures 'properties' dict exists
    - Extracts 'required' array to root level
    - Sets 'additionalProperties' to False for strict validation
    - Recursively cleans nested objects
    """
    schema = copy.deepcopy(pydantic_schema)

    # Handle allOf structures (Pydantic sometimes uses these for inheritance)
    if 'allOf' in schema:
        # Merge all schemas in allOf
        merged = {}
        for sub_schema in schema['allOf']:
            if '$ref' in sub_schema:
                continue  # Skip references, they're usually in definitions
            merged.update(sub_schema)
        schema = merged

    # Remove incompatible fields
    fields_to_remove = ['title', 'definitions', '$ref', '$schema']
    for field in fields_to_remove:
        schema.pop(field, None)

    # Ensure required fields exist
    if 'type' not in schema:
        schema['type'] = 'object'

    if 'properties' not in schema:
        schema['properties'] = {}

    # Extract required fields (may be nested in allOf)
    if 'required' not in schema:
        schema['required'] = []

    # Set additionalProperties to False for strict validation
    schema['additionalProperties'] = False

    # Recursively clean nested properties
    for prop_name, prop_schema in schema.get('properties', {}).items():
        if isinstance(prop_schema, dict):
            # Recursively clean nested objects
            if prop_schema.get('type') == 'object':
                schema['properties'][prop_name] = _clean_pydantic_schema(prop_schema)
            # Clean arrays with object items
            elif prop_schema.get('type') == 'array':
                items = prop_schema.get('items', {})
                if isinstance(items, dict) and items.get('type') == 'object':
                    schema['properties'][prop_name]['items'] = _clean_pydantic_schema(items)
                # Remove problematic fields from items
                for field in fields_to_remove:
                    items.pop(field, None)
            # Remove problematic fields from simple types
            else:
                for field in fields_to_remove:
                    prop_schema.pop(field, None)

    return schema


def _create_error_wrapped_function(tool_name: str, original_func: Callable) -> Callable:
    """
    Wrap a tool function with error handling.

    Args:
        tool_name: Name of the tool (for error messages)
        original_func: The original tool function to wrap

    Returns:
        Wrapped function that catches and formats errors
    """
    def wrapped_func(**kwargs) -> str:
        """Execute tool with error handling."""
        try:
            result = original_func(**kwargs)
            # Convert result to string for consistency
            return str(result)
        except Exception as e:
            error_msg = f"Error in tool '{tool_name}': {type(e).__name__}: {str(e)}"
            print(f"[Tool Error] {error_msg}")
            return error_msg

    # Preserve function metadata
    wrapped_func.__name__ = original_func.__name__
    wrapped_func.__doc__ = original_func.__doc__

    return wrapped_func


def convert_langchain_tools(
    langchain_tools: List[Any]
) -> Tuple[List[Dict], Dict[str, Callable]]:
    """
    Convert LangChain BaseTool instances to OpenAI-compatible format.

    This function extracts tool schemas and callables from LangChain tools,
    making them compatible with OpenAI function calling and the agent framework.

    Args:
        langchain_tools: List of LangChain BaseTool instances
                        (e.g., from PlayWrightBrowserToolkit.get_tools())

    Returns:
        Tuple of (tool_schemas, tool_functions):
            - tool_schemas: List of OpenAI-compatible tool schema dicts
            - tool_functions: Dict mapping tool names to wrapped callables

    Example:
        >>> from langchain_community.agent_toolkits.playwright.toolkit import PlayWrightBrowserToolkit
        >>> toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
        >>> lc_tools = toolkit.get_tools()
        >>> schemas, functions = convert_langchain_tools(lc_tools)
        >>> # Use schemas with OpenAI API, functions for execution

    Raises:
        AttributeError: If tools missing required attributes (name, description, args_schema, _run)
        ValueError: If schema conversion fails
    """
    tool_schemas = []
    tool_functions = {}

    for tool in langchain_tools:
        try:
            # Extract tool metadata
            tool_name = tool.name
            tool_description = tool.description or f"Tool: {tool_name}"

            # Get Pydantic schema from args_schema
            if hasattr(tool, 'args_schema') and tool.args_schema is not None:
                # Get schema from Pydantic model
                pydantic_schema = tool.args_schema.schema()
            else:
                # Fallback: tool has no args
                pydantic_schema = {
                    "type": "object",
                    "properties": {},
                    "required": []
                }

            # Clean schema for OpenAI compatibility
            cleaned_params = _clean_pydantic_schema(pydantic_schema)

            # Create OpenAI-compatible tool schema
            tool_schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": tool_description,
                    "parameters": cleaned_params
                }
            }

            tool_schemas.append(tool_schema)

            # Extract callable (use _run for sync execution)
            if hasattr(tool, '_run'):
                original_func = tool._run
            elif hasattr(tool, 'run'):
                original_func = tool.run
            else:
                raise AttributeError(f"Tool '{tool_name}' has no _run or run method")

            # Wrap with error handling
            wrapped_func = _create_error_wrapped_function(tool_name, original_func)
            tool_functions[tool_name] = wrapped_func

        except Exception as e:
            print(f"[Warning] Failed to convert tool: {e}")
            # Continue processing other tools
            continue

    return tool_schemas, tool_functions


def convert_langchain_toolkit(toolkit: Any) -> Tuple[List[Dict], Dict[str, Callable]]:
    """
    Convenience function to convert an entire LangChain toolkit.

    Args:
        toolkit: LangChain toolkit instance with get_tools() method

    Returns:
        Tuple of (tool_schemas, tool_functions)

    Example:
        >>> toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=browser)
        >>> schemas, functions = convert_langchain_toolkit(toolkit)
    """
    if not hasattr(toolkit, 'get_tools'):
        raise ValueError("Toolkit must have a get_tools() method")

    langchain_tools = toolkit.get_tools()
    return convert_langchain_tools(langchain_tools)

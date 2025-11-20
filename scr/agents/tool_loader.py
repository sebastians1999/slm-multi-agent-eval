"""
Tool Loader

Initializes and registers all available tool categories at import time.
This module serves as the central configuration point for all agent tools.

Tool Categories:
- 'search': Web search capabilities (Tavily)
- 'code': Python code execution (E2B Sandbox)
- 'browser': Web browser navigation (PlayWright via LangChain)

Usage:
    Simply import this module to trigger registration:
    >>> from .tool_loader import tool_registry
    >>> # All tools are now registered

    Or import specific utilities:
    >>> from .tool_loader import get_tools_for_agent
    >>> tools, functions = get_tools_for_agent(['search', 'code'])
"""

from typing import List, Dict, Callable, Tuple, Optional
from .utils.tool_registry import tool_registry


def _register_search_tools():
    """Register search tool category."""
    try:
        from .tools.search import tavily_search, SEARCH_TOOL_SCHEMA

        tool_registry.register_category(
            category='search',
            tools=[SEARCH_TOOL_SCHEMA],
            tool_functions={'tavily_search': tavily_search}
        )
        print("[Tool Loader] Registered 'search' category")
    except Exception as e:
        print(f"[Tool Loader] Warning: Failed to register search tools: {e}")


def _register_code_tools():
    """Register code execution tool category."""
    try:
        from .tools.python_interpreter import execute_python, PYTHON_TOOL_SCHEMA

        tool_registry.register_category(
            category='code',
            tools=[PYTHON_TOOL_SCHEMA],
            tool_functions={'execute_python': execute_python}
        )
        print("[Tool Loader] Registered 'code' category")
    except Exception as e:
        print(f"[Tool Loader] Warning: Failed to register code tools: {e}")


def _register_browser_tools():
    """Register browser navigation tool category."""
    try:
        from .tools.browser import get_playwright_tools

        # Initialize PlayWright tools (returns schemas, functions, and manager)
        browser_schemas, browser_functions, browser_manager = get_playwright_tools()

        # Register with metadata containing browser manager for cleanup
        tool_registry.register_category(
            category='browser',
            tools=browser_schemas,
            tool_functions=browser_functions,
            metadata={'browser_manager': browser_manager}
        )
        print("[Tool Loader] Registered 'browser' category")
    except ImportError as e:
        print(f"[Tool Loader] Info: Browser tools not available (dependencies missing): {e}")
        print("[Tool Loader] To enable browser tools, run:")
        print("  pip install langchain-community playwright")
        print("  playwright install chromium")
    except Exception as e:
        print(f"[Tool Loader] Warning: Failed to register browser tools: {e}")


def _initialize_all_categories():
    """
    Initialize and register all available tool categories.

    Called automatically on module import. Categories are registered
    in order: search, code, browser.
    """
    _register_search_tools()
    _register_code_tools()
    _register_browser_tools()


# Auto-register all categories on import
_initialize_all_categories()


# Utility functions for easy access
def get_tools_for_agent(
    categories: Optional[List[str]] = None
) -> Tuple[List[Dict], Dict[str, Callable]]:
    """
    Get tools and functions for specified categories.

    Convenience wrapper around tool_registry.get_tools_by_categories().

    Args:
        categories: List of category names, or None for all categories

    Returns:
        Tuple of (tool_schemas, tool_functions)

    Example:
        >>> # Get all tools
        >>> all_tools, all_functions = get_tools_for_agent()
        >>>
        >>> # Get specific categories
        >>> tools, functions = get_tools_for_agent(['search', 'browser'])
    """
    return tool_registry.get_tools_by_categories(categories)


def get_browser_manager():
    """
    Get the browser manager instance if browser tools are registered.

    Returns:
        BrowserManager instance or None if browser tools not available

    Example:
        >>> manager = get_browser_manager()
        >>> if manager:
        ...     # Browser tools are available
        ...     manager.close()  # Cleanup when done
    """
    if tool_registry.category_exists('browser'):
        metadata = tool_registry.get_category_metadata('browser')
        return metadata.get('browser_manager')
    return None


def list_available_categories() -> List[str]:
    """
    List all registered tool categories.

    Returns:
        List of category names

    Example:
        >>> categories = list_available_categories()
        >>> print(f"Available: {categories}")
        Available: ['search', 'code', 'browser']
    """
    return tool_registry.list_categories()

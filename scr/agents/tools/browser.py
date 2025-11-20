"""
PlayWright Browser Tools Integration

Integrates LangChain PlayWrightBrowserToolkit into the agent framework
using the generic LangChain adapter and browser lifecycle manager.
"""

from typing import Dict, List, Callable, Tuple
from ..utils.browser_manager import BrowserManager
from ..utils.langchain_adapter import convert_langchain_toolkit


def get_playwright_tools() -> Tuple[List[Dict], Dict[str, Callable], BrowserManager]:
    """
    Initialize and return PlayWright browser tools.

    This function creates a browser manager, initializes the LangChain
    PlayWrightBrowserToolkit, and converts it to OpenAI-compatible format.

    Returns:
        Tuple of (tool_schemas, tool_functions, browser_manager):
            - tool_schemas: List of OpenAI-compatible tool schemas
            - tool_functions: Dict mapping tool names to callables
            - browser_manager: BrowserManager instance for cleanup

    Raises:
        ImportError: If langchain-community or playwright not installed
        RuntimeError: If browser initialization fails

    Example:
        >>> schemas, functions, manager = get_playwright_tools()
        >>> # Use schemas with OpenAI API, functions for execution
        >>> # Cleanup when done:
        >>> manager.close()

    Available Tools (from PlayWrightBrowserToolkit):
        - navigate_browser: Navigate to a URL
        - navigate_back: Go back in browser history
        - navigate_forward: Go forward in browser history
        - click_element: Click on an element
        - extract_text: Extract text from the current page
        - extract_hyperlinks: Extract all links from the page
        - get_elements: Get elements matching a selector
        - current_url: Get the current page URL
    """
    try:
        from langchain_community.agent_toolkits.playwright.toolkit import (
            PlayWrightBrowserToolkit
        )
        from langchain_community.tools.playwright.utils import (
            create_sync_playwright_browser
        )
    except ImportError as e:
        raise ImportError(
            "LangChain PlayWright dependencies not installed. Install with:\n"
            "  pip install langchain-community playwright\n"
            "  playwright install chromium"
        ) from e

    # Create browser manager (lazy initialization)
    browser_manager = BrowserManager()

    # Get browser instance (triggers initialization)
    try:
        sync_browser = browser_manager.get_browser()
    except Exception as e:
        browser_manager.close()
        raise RuntimeError(f"Failed to initialize browser: {e}") from e

    # Initialize PlayWright toolkit with the browser
    try:
        toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=sync_browser)
    except Exception as e:
        browser_manager.close()
        raise RuntimeError(f"Failed to create PlayWright toolkit: {e}") from e

    # Convert LangChain tools to OpenAI format
    try:
        tool_schemas, tool_functions = convert_langchain_toolkit(toolkit)
    except Exception as e:
        browser_manager.close()
        raise RuntimeError(f"Failed to convert browser tools: {e}") from e

    print(f"[Browser Tools] Initialized {len(tool_schemas)} PlayWright tools")

    return tool_schemas, tool_functions, browser_manager

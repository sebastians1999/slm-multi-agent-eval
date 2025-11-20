"""
Agent utilities package.

This package contains utility functions and classes for agents:
- helpers: General utility functions (extract_final_answer, etc.)
- tool_registry: Category-based tool registry
- langchain_adapter: Convert LangChain tools to OpenAI format
- browser_manager: Manage Playwright browser lifecycle
"""

# Import from submodules for convenient access
from .helpers import extract_final_answer
from .tool_registry import tool_registry, ToolRegistry
from .langchain_adapter import convert_langchain_tools, convert_langchain_toolkit
from .browser_manager import BrowserManager

__all__ = [
    # Helper functions
    'extract_final_answer',
    # Tool management
    'tool_registry',
    'ToolRegistry',
    # LangChain integration
    'convert_langchain_tools',
    'convert_langchain_toolkit',
    # Browser management
    'BrowserManager',
]

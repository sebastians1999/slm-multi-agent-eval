from langchain_community.agent_toolkits import PlayWrightBrowserToolkit
from langchain_community.tools.playwright.utils import create_sync_playwright_browser
from typing import Dict, Any
import json


_browser = None
_toolkit = None
_tools_by_name = None


def _initialize_browser():
    """Initialize the browser and toolkit once."""
    global _browser, _toolkit, _tools_by_name

    if _browser is None:
        _browser = create_sync_playwright_browser()
        _toolkit = PlayWrightBrowserToolkit.from_browser(sync_browser=_browser)
        tools = _toolkit.get_tools()
        _tools_by_name = {tool.name: tool for tool in tools}


def navigate_browser(url: str) -> str:
    """Navigate to a URL."""
    _initialize_browser()
    tool = _tools_by_name["navigate_browser"]
    return tool.run({"url": url})


def extract_text() -> str:
    """Extract all text from the current webpage."""
    _initialize_browser()
    tool = _tools_by_name["extract_text"]
    return tool.run({})


def extract_hyperlinks() -> str:
    """Extract all hyperlinks from the current webpage."""
    _initialize_browser()
    tool = _tools_by_name["extract_hyperlinks"]
    return tool.run({})


def get_current_page() -> str:
    """Get the current page URL."""
    _initialize_browser()
    tool = _tools_by_name["current_webpage"]
    return tool.run({})


def click_element(selector: str) -> str:
    """Click on an element specified by CSS selector."""
    _initialize_browser()
    tool = _tools_by_name["click_element"]
    return tool.run({"selector": selector})


def get_elements(selector: str, attributes: list = None) -> str:
    """Get elements by CSS selector with optional attributes."""
    _initialize_browser()
    tool = _tools_by_name["get_elements"]

    if attributes is None:
        attributes = ["innerText"]

    result = tool.run({"selector": selector, "attributes": attributes})
    return result

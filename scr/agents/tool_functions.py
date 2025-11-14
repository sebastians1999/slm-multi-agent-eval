from .tools.search import tavily_search
from .tools.python_interpreter import execute_python
from .tools.browser import navigate_browser, extract_text, get_current_page

tool_functions = {
    "tavily_search": tavily_search,
    "execute_python": execute_python,
    # "navigate_browser": navigate_browser,
    # "extract_text": extract_text,
    # "get_current_page": get_current_page
}

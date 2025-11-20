# LangChain PlayWright Browser Tools Integration

## Summary

Successfully integrated LangChain PlayWrightBrowserToolkit with dynamic tool loading by category. This implementation provides a flexible, plug-and-play architecture for adding any LangChain toolkit to your agents.

## What Was Implemented

### 1. **Tool Registry System** (`scr/agents/utils/tool_registry.py`)
- Centralized registry for organizing tools by category
- Singleton pattern for global access
- Categories: `'search'`, `'code'`, `'browser'`
- Metadata support for tool-specific resources (e.g., browser manager)

### 2. **Generic LangChain Adapter** (`scr/agents/utils/langchain_adapter.py`)
- Converts any LangChain `BaseTool` to OpenAI-compatible format
- Cleans Pydantic schemas for compatibility
- Wraps tool functions with error handling
- Reusable for future LangChain toolkits

### 3. **Browser Lifecycle Manager** (`scr/agents/utils/browser_manager.py`)
- Manages shared Playwright browser instance
- Lazy initialization (browser created only when needed)
- Automatic cleanup and crash recovery
- Context manager support

### 4. **PlayWright Tools Integration** (`scr/agents/tools/browser.py`)
- Integrates LangChain PlayWrightBrowserToolkit
- Uses browser manager for efficient reuse
- Provides ~8 browser navigation tools automatically

### 5. **Dynamic Tool Loading** (Updated `BaseAgent`)
- New `tool_categories` parameter for selective tool loading
- Default behavior: loads all tools when no categories specified
- Automatic browser cleanup when agent is destroyed
- Context manager support for resource cleanup

### 6. **Updated Tool Modules**
- `scr/agents/tools/search.py`: Added `SEARCH_TOOL_SCHEMA` constant
- `scr/agents/tools/python_interpreter.py`: Added `PYTHON_TOOL_SCHEMA` constant

### 7. **Tool Loader** (`scr/agents/tool_loader.py`)
- Registers all tool categories at import time
- Graceful handling of missing dependencies
- Utility functions for easy tool access

## Installation

### 1. Install Python Dependencies
```bash
uv sync
```

### 2. Install Playwright Browser (Required for browser tools)
```bash
bash scripts/setup_browser.sh
```

Or manually:
```bash
playwright install chromium
```

## Usage

### Basic: Load All Tools (Default)
```python
from scr.agents.base_agent import BaseAgent

# Loads all available tools including browser
agent = BaseAgent(model="gpt-4", temperature=0.3)

# Use agent...
response = agent.invoke([{"role": "user", "content": "Search for Python tutorials"}])

# Cleanup
agent.cleanup()
```

### With Context Manager (Recommended)
```python
with BaseAgent(model="gpt-4") as agent:
    response = agent.invoke([{"role": "user", "content": "Navigate to example.com"}])
    # Automatic cleanup when exiting context
```

### Load Specific Tool Categories
```python
# Only search tools
agent = BaseAgent(model="gpt-4", tool_categories=['search'])

# Search + code execution
agent = BaseAgent(model="gpt-4", tool_categories=['search', 'code'])

# Only browser tools
agent = BaseAgent(model="gpt-4", tool_categories=['browser'])

# Multiple categories
agent = BaseAgent(model="gpt-4", tool_categories=['search', 'browser', 'code'])
```

### Use Custom Tools (for Testing)
```python
# Define custom tool
custom_tool = {
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "My custom tool",
        "parameters": {...}
    }
}

def my_tool_function(**kwargs):
    return "result"

# Use custom tools (bypasses category system)
agent = BaseAgent(
    model="gpt-4",
    tools=[custom_tool],
    tool_functions={"my_tool": my_tool_function}
)
```

### Check Available Tools
```python
from scr.agents.tool_loader import list_available_categories

categories = list_available_categories()
print(f"Available categories: {categories}")
# Output: ['search', 'code', 'browser']
```

## Available Tool Categories

### `'search'` - Web Search
- **Tool**: `tavily_search`
- **Description**: Search the web with Tavily API
- **Dependencies**: `tavily-python` (already installed)
- **Required**: `TAVILY_API_KEY` in `.env`

### `'code'` - Python Code Execution
- **Tool**: `execute_python`
- **Description**: Execute Python code in secure E2B sandbox
- **Dependencies**: `e2b-code-interpreter` (already installed)
- **Required**: `E2B_API_KEY` in `.env`

### `'browser'` - Web Browser Navigation
- **Tools** (from PlayWrightBrowserToolkit):
  - `navigate_browser` - Navigate to a URL
  - `navigate_back` - Go back in browser history
  - `navigate_forward` - Go forward in browser history
  - `click_element` - Click on an element
  - `extract_text` - Extract text from current page
  - `extract_hyperlinks` - Extract all links
  - `get_elements` - Get elements matching a selector
  - `current_url` - Get current page URL
- **Dependencies**: `langchain-community`, `playwright`
- **Setup**: Run `bash scripts/setup_browser.sh` after installing dependencies

## Architecture Benefits

### ✅ Plug-and-Play
Add new LangChain toolkits with minimal code:
```python
from langchain_community.agent_toolkits import SomeToolkit
from scr.agents.utils.langchain_adapter import convert_langchain_toolkit

toolkit = SomeToolkit.from_config(...)
schemas, functions = convert_langchain_toolkit(toolkit)
tool_registry.register_category('new_category', schemas, functions)
```

### ✅ Efficient Resource Management
- Browser instance shared across all tool calls
- Automatic cleanup prevents resource leaks
- Lazy initialization reduces overhead

### ✅ Flexible Configuration
- Load only the tools you need
- Reduce API payload size with selective loading
- Category-based organization keeps tools organized

### ✅ Robust Error Handling
- Schema validation and cleaning
- Wrapped tool execution with error capture
- Graceful degradation if dependencies missing

## File Structure

```
scr/agents/
├── base_agent.py              # Updated with dynamic tool loading
├── tool_loader.py             # Registers all tool categories
├── utils/
│   ├── tool_registry.py       # Category-based tool registry
│   ├── langchain_adapter.py   # LangChain → OpenAI converter
│   └── browser_manager.py     # Browser lifecycle management
└── tools/
    ├── search.py              # Tavily search + schema
    ├── python_interpreter.py  # E2B execution + schema
    └── browser.py             # PlayWright tools integration

scripts/
└── setup_browser.sh           # Browser installation script

test/
├── test_tool_registry.py      # Registry tests
├── test_browser_manager.py    # Browser lifecycle tests
├── test_langchain_adapter.py  # Adapter tests
└── test_agent_tool_categories.py  # Integration tests
```

## Removed Files
- ~~`scr/agents/tool_functions.py`~~ (replaced by `tool_loader.py`)
- ~~`scr/agents/tools_list.py`~~ (schemas now in individual tool modules)

## Testing

Run all tests:
```bash
pytest test/
```

Run specific test suites:
```bash
pytest test/test_tool_registry.py        # Tool registry tests
pytest test/test_langchain_adapter.py    # Adapter tests
pytest test/test_browser_manager.py      # Browser manager tests
pytest test/test_agent_tool_categories.py  # Integration tests
```

**Note**: Some browser tests are skipped by default as they require playwright browser binaries.

## Adding New Tool Categories

### Example: Adding a New LangChain Toolkit

1. **Create integration module** (`scr/agents/tools/new_toolkit.py`):
```python
from langchain_community.agent_toolkits import NewToolkit
from ..utils.langchain_adapter import convert_langchain_toolkit

def get_new_toolkit_tools():
    toolkit = NewToolkit.from_config(...)
    schemas, functions = convert_langchain_toolkit(toolkit)
    return schemas, functions
```

2. **Register in tool loader** (`scr/agents/tool_loader.py`):
```python
def _register_new_category():
    from .tools.new_toolkit import get_new_toolkit_tools
    schemas, functions = get_new_toolkit_tools()
    tool_registry.register_category('new_category', schemas, functions)

# Add to initialization
_initialize_all_categories():
    _register_search_tools()
    _register_code_tools()
    _register_browser_tools()
    _register_new_category()  # Add here
```

3. **Use it**:
```python
agent = BaseAgent(model="gpt-4", tool_categories=['new_category'])
```

## Migration Guide

### Before (Old Approach)
```python
from scr.agents.base_agent import BaseAgent
from scr.agents.tools_list import tools
from scr.agents.tool_functions import tool_functions

# All tools loaded by default, no control
agent = BaseAgent(model="gpt-4", tools=tools, tool_functions=tool_functions)
```

### After (New Approach)
```python
from scr.agents.base_agent import BaseAgent

# Load all tools (backward compatible)
agent = BaseAgent(model="gpt-4")

# Or load specific categories
agent = BaseAgent(model="gpt-4", tool_categories=['search', 'browser'])
```

## Troubleshooting

### Browser Tools Not Loading
```
[Tool Loader] Info: Browser tools not available (dependencies missing)
```
**Solution**: Run `bash scripts/setup_browser.sh` or manually:
```bash
uv sync
playwright install chromium
```

### Import Errors
```
ImportError: No module named 'playwright'
```
**Solution**: Install dependencies:
```bash
uv sync
```

### Browser Fails to Initialize
```
RuntimeError: Failed to initialize browser
```
**Solutions**:
1. Ensure browser binaries installed: `playwright install chromium`
2. Check system compatibility (headless mode requires certain libraries on Linux)
3. Try installing system dependencies: `playwright install-deps`

## Performance Considerations

- **Browser overhead**: Browser initialization takes ~1-2 seconds. Use `tool_categories` to exclude browser if not needed.
- **Shared browser**: Browser instance reused across calls for efficiency
- **Lazy loading**: Browser only created when first tool call is made
- **Memory**: Browser uses ~100-200MB RAM when active

## Security Considerations

⚠️ **Warning**: Browser tools can navigate to any URL and execute JavaScript. Use appropriate sandboxing:
- Run in isolated environments for untrusted inputs
- Implement URL whitelisting if needed
- Monitor and log browser activities
- Set timeouts to prevent hanging operations

## Future Enhancements

Potential improvements for future iterations:
- [ ] Async tool support (use `tool._arun`)
- [ ] Tool-specific timeout configuration
- [ ] Browser context pooling for parallel operations
- [ ] Custom browser configuration (viewport, user-agent, etc.)
- [ ] Tool usage metrics and analytics
- [ ] Dynamic category loading/unloading at runtime

## Contributing

When adding new tool categories:
1. Create tool module in `scr/agents/tools/`
2. Add schema constant in tool module
3. Register in `tool_loader.py`
4. Add tests in `test/`
5. Update this documentation

## Questions?

For issues or questions about this integration, check:
- Test files for usage examples
- Tool loader for registration patterns
- LangChain adapter for conversion logic

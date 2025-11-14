tools = [
    {
            "type": "function",
            "function": {
                "name": "tavily_search",
                "description": "Search the web with Tavily for up-to-date information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                        "max_results": {"type": "integer", "default": 5},
                    },
                    "required": ["query"],
                },
            },
        },
    {
        "type": "function",
        "function": {
            "name": "execute_python",
            "description": "Execute python code in a Jupyter notebook cell and return result",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "The python code to execute in a single cell"
                    }
                },
                "required": ["code"]
            }
        }
    },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "navigate_browser",
    #         "description": "Navigate to a URL in the browser",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "url": {"type": "string", "description": "The URL to navigate to"}
    #             },
    #             "required": ["url"]
    #         }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "extract_text",
    #         "description": "Extract all text from the current webpage",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {}
    #         }
    #     }
    # },
    # {
    #     "type": "function",
    #     "function": {
    #         "name": "get_current_page",
    #         "description": "Get the URL of the current webpage",
    #         "parameters": {
    #             "type": "object",
    #             "properties": {}
    #         }
    #     }
    # }
]
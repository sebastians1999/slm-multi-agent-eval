from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

tavily_client = TavilyClient(api_key=api_key)

# OpenAI-compatible tool schema
SEARCH_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "tavily_search",
        "description": "Search the web with Tavily for up-to-date information. Returns search results with snippets and URLs.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up"
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of search results to return",
                    "default": 5
                },
            },
            "required": ["query"],
            "additionalProperties": False
        },
    },
}

def tavily_search(**kwargs):
    """
    Search the web using Tavily API.
    Args:
        query (str): The search query
        max_results (int, optional): Maximum number of results. Defaults to 5.
        **kwargs: Additional arguments passed to Tavily client
    Returns:
        Search results from Tavily API
    """
    
    kwargs["search_depth"] = "advanced"
    #kwargs["include_raw_content"] = "markdown"
    
    print("Tavily used.")
    return tavily_client.search(**kwargs)
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

tavily_client = TavilyClient(api_key=api_key)

def tavily_search(**kwargs):
    
    print("Tavily used!")
    
    return tavily_client.search(**kwargs)
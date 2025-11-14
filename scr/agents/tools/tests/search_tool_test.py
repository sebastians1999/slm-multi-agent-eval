from tavily import TavilyClient
from dotenv import load_dotenv
import os



if __name__ == "__main__": 
    
    
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable is not set")
    
    tavily_client = TavilyClient(api_key=api_key)
    response = tavily_client.search("Who is Leo Messi?", search_depth="advanced")
    
    print(response)
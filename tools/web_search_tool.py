from langchain.tools import Tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os

load_dotenv()

# initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def web_search(query: str) -> str:
    """web search for latest information"""
    try:
        # search
        results = tavily_client.search(query, max_results=5)
        
        # format results
        formatted_results = []
        for i, result in enumerate(results.get('results', []), 1):
            formatted_results.append(
                f"{i}. {result.get('title', 'No title')}\n"
                f"   link: {result.get('url', 'No URL')}\n"
                f"   summary: {result.get('content', 'No content')[:200]}..."
            )
        
        if formatted_results:
            return "\n\n".join(formatted_results)
        else:
            return "No related search results found"
    
    except Exception as e:
        return f"Search error: {str(e)}"

# create tool
WebSearchTool = Tool(
    name="WebSearchTool",
    func=web_search,
    description="Real-time web search for current information, providing evidence for other tools"
)
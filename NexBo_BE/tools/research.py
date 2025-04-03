from langchain.tools import tool
from langchain_community.tools import TavilySearchResults
from dotenv import load_dotenv
import os

load_dotenv()  # Loads environment variables, e.g., TAVILY_API_KEY
T_KEY = os.getenv("TAVILY_API_KEY")

@tool
def tavily_search(query: str) -> str:
    """
    Search the web using the Tavily Search API and return formatted search results.

    Args:
        query (str): The search query string.

    Returns:
        str: A formatted string with search results.
    """
    try:
        # Create an instance of the TavilySearchResults tool with desired configuration.
        search_tool = TavilySearchResults(
            max_results=5,
            search_depth="basic",
            include_answer=True,
            include_raw_content=True,
            include_images=True,
        )

        # Execute the search.
        raw_results = search_tool.run(query)

        # Format the results into a neat string.
        # If the raw output is a list or dict, we attempt to format each result.
        if isinstance(raw_results, list):
            results = raw_results
        elif isinstance(raw_results, dict) and "results" in raw_results:
            results = raw_results["results"]
        else:
            # Fallback: return the string representation
            return str(raw_results)

        formatted_results = []
        for idx, item in enumerate(results, start=1):
            title = item.get("title", "No Title")
            url = item.get("url") or item.get("link", "No URL")
            snippet = item.get("content", "No snippet available.")
            formatted_results.append(
                f"{idx}. {title}\nURL: {url}\nSnippet: {snippet}\n"
            )
        return "\n".join(formatted_results)

    except Exception as e:
        return f"⚠️ Error retrieving search results: {str(e)}"

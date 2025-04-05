from Recomd.Rec import StockRecommendationEngine,explain_recommendation_tool
from langchain.tools import StructuredTool
from typing import Optional, Dict, List

def get_recommendations_tool(budget, market='US', currency='USD', sector=None, criteria=None, limit=5):
    """
    Recommends stocks based on a specified budget, market, sector, and additional criteria.

    Parameters:
        budget (number): The maximum amount available for investing in a single stock.
        market (str): The target stock market to search in. Use 'US' for United States or 'IN' for India.
        currency (str): The currency in which stock prices are represented (e.g., 'USD' or 'INR').
        sector (str, optional): The industry sector for which to fetch stock recommendations (e.g., 'Technology').
        criteria (dict, optional): Additional filters to refine recommendations (e.g., risk tolerance, market cap).
        limit (int): The maximum number of stock recommendations to return. Defaults to 5.

    Returns:
        list: A list of recommended stocks that match the given parameters.
    """
    engine = StockRecommendationEngine()
    recommendations = engine.get_recommendations(budget, market, currency, sector, criteria, limit)
    engine.close()
    return recommendations

def explain_stock_recommended(ticker: str) -> str:
    """
    Provides an explanation for why a specific stock was recommended.

    Parameters:
        ticker (str): The stock ticker symbol.

    Returns:
        str: A detailed explanation for the recommendation.
    """
    explanation = explain_recommendation_tool(ticker)
    return explanation

# Define the structured tool for stock recommendations.
recommend_stock_tool = StructuredTool.from_function(
    func=get_recommendations_tool,
    name="recommend_stock",
    description=(
        "Recommends stocks based on investment criteria including budget, market, currency, sector, "
        "and additional filtering options. Use this tool to get tailored stock recommendations."
    ),
    args_schema={
        "type": "object",
        "properties": {
            "budget": {
                "type": "number",
                "description": "The maximum amount available for investing in a single stock."
            },
            "market": {
                "type": "string",
                "description": "Stock market to search in (e.g., 'US' for United States or 'IN' for India).",
                "default": "US"
            },
            "currency": {
                "type": "string",
                "description": "Currency for stock prices (e.g., 'USD' or 'INR').",
                "default": None
            },
            "sector": {
                "type": "string",
                "description": "The sector in which to search for stock recommendations (e.g., 'Technology', 'Healthcare').",
                "default": None
            },
            "criteria": {
                "type": "object",
                "description": "Additional filtering criteria (e.g., risk tolerance, market cap).",
                "default": None
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of stock recommendations to return.",
                "default": 5
            }
        },
        "required": ["budget"]
    }
)
# import json
# recommendations= get_recommendations_tool(market='US',currency='USD',budget=20000,sector='Healthcare',limit=5)
# rec = json.dumps(recommendations, indent=4)
# print(rec)

explain_stock_tool = StructuredTool.from_function(
    func=explain_stock_recommended,
    name="explain_stock_recommended",
    description=(
        "Provides a detailed explanation for why a specific stock was recommended. "
        "This explanation includes factors such as the stock's performance, risk profile, "
        "financial metrics, and how it aligns with the user's investment criteria."
    ),
    args_schema={
        "type": "object",
        "properties": {
            "ticker": {
                "type": "string",
                "description": "The stock ticker symbol for which you need an explanation."
            }
        },
        "required": ["ticker"]
    }
)

tools = [recommend_stock_tool, explain_stock_tool]

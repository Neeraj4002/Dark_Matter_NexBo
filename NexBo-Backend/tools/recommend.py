from Recomd.Rec import StockRecommendationEngine, explain_recommendation_tool
from langchain.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Dict

# ✅ Define Pydantic schema for stock recommendations
class RecommendStockSchema(BaseModel):
    budget: float = Field(..., description="The maximum amount available for investing in a single stock.")
    market: str = Field(default="US", description="Stock market to search in (e.g., 'US' or 'IN').")
    currency: Optional[str] = Field(default="USD", description="Currency for stock prices (e.g., 'USD' or 'INR').")
    sector: Optional[str] = Field(default=None, description="The sector to search for stock recommendations.")
    criteria: Optional[Dict] = Field(default=None, description="Additional filtering criteria (e.g., risk tolerance).")
    limit: int = Field(default=5, description="Maximum number of stock recommendations to return.")

# ✅ Define function to get stock recommendations
def get_recommendations_tool(budget: float, market: str = 'US', currency: str = 'USD', 
                             sector: Optional[str] = None, criteria: Optional[Dict] = None, limit: int = 5):
    """
    Recommends stocks based on a specified budget, market, sector, and additional criteria.

    Parameters:
        budget (float): The maximum amount available for investing in a single stock.
        market (str): The target stock market to search in. Default is 'US'.
        currency (str): The currency in which stock prices are represented (e.g., 'USD' or 'INR').
        sector (str, optional): The industry sector for which to fetch stock recommendations.
        criteria (dict, optional): Additional filters to refine recommendations (e.g., risk tolerance).
        limit (int): The maximum number of stock recommendations to return. Defaults to 5.

    Returns:
        list: A list of recommended stocks that match the given parameters.
    """
    engine = StockRecommendationEngine()
    recommendations = engine.get_recommendations(budget, market, currency, sector, criteria, limit)
    engine.close()
    return recommendations

# ✅ Define Pydantic schema for explaining stock recommendations
class ExplainStockSchema(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol for which you need an explanation.")

# ✅ Define function to explain why a stock was recommended
def explain_stock_recommended(ticker: str) -> str:
    """
    Provides an explanation for why a specific stock was recommended.

    Parameters:
        ticker (str): The stock ticker symbol.

    Returns:
        str: A detailed explanation for the recommendation.
    """
    return explain_recommendation_tool(ticker)

# ✅ Structured tool for stock recommendations
recommend_stock_tool = StructuredTool.from_function(
    func=get_recommendations_tool,
    name="recommend_stock",
    description="Recommends stocks based on investment criteria including budget, market, currency, sector, and additional filters.",
    args_schema=RecommendStockSchema  # ✅ Correctly using Pydantic schema
)

# ✅ Structured tool for explaining stock recommendations
explain_stock_tool = StructuredTool.from_function(
    func=explain_stock_recommended,
    name="explain_stock_recommended",
    description="Provides an explanation for why a specific stock was recommended.",
    args_schema=ExplainStockSchema  # ✅ Correctly using Pydantic schema
)

# ✅ List of available tools
tools = [recommend_stock_tool, explain_stock_tool]

# # ✅ Test (optional)
# if __name__ == "__main__":
#     import json
#     # Example test call
#     recommendations = get_recommendations_tool(budget=20000, market='US', currency='USD', sector='Healthcare', limit=5)
#     print(json.dumps(recommendations, indent=4))

#     explanation = explain_stock_recommended(ticker="AAPL")
#     print(explanation)

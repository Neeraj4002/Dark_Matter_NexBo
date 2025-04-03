from langchain.tools import tool
from yah.yah import get_stock


@tool
def stock_lookup(ticker: str) -> str:
    """
    Fetch stock details such as current price, market cap, and P/E ratio using the Yahoo Finance API.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL").

    Returns:
        str: A formatted string with stock information.
    """
    try:
        stock_data = get_stock(ticker)
        if not stock_data:
            return f"Stock data for {ticker} not found."
        return (
            f"📈 **Stock: {stock_data['name']} ({stock_data['symbol']})**\n"
            f"💰 **Current Price:** ${stock_data['current_price']}\n"
            f"🏢 **Market Cap:** {stock_data['market_cap']}\n"
            f"📊 **52-Week High/Low:** {stock_data['52_week_high']}/{stock_data['52_week_low']}\n"
            f"📉 **P/E Ratio:** {stock_data['pe_ratio']}\n"
        )
    except Exception as e:
        return f"⚠️ Error retrieving stock data: {str(e)}"



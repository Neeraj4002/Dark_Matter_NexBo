import yfinance as yf

def get_stock(ticker):
    """Fetch stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "symbol": ticker.upper(),
            "name": info.get("shortName", "N/A"),
            "current_price": info.get("regularMarketPrice", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "pe_ratio": info.get("trailingPE", "N/A"),
            "52_week_high": info.get("fiftyTwoWeekHigh", "N/A"),
            "52_week_low": info.get("fiftyTwoWeekLow", "N/A"),
        }
    except Exception as e:
        print(f"⚠️ Error fetching stock data: {e}")
        return None


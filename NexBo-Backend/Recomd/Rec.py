# =======================
# 1. Imports and Setup
# =======================
import sqlite3
import threading
from datetime import datetime
import yfinance as yf

# =======================
# 2. Database and Engine Class Definition
# =======================
class StockRecommendationEngine:
    def __init__(self, db_path='E:/NexBO/stock_database.db'):
        self.db_path = db_path
        # Use check_same_thread=False to allow multithreading with SQLite
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.setup_database()

    def setup_database(self):
        """Create the stocks table if it doesn't exist."""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS stocks (
                ticker TEXT PRIMARY KEY,
                company_name TEXT,
                sector TEXT,
                industry TEXT,
                current_price REAL,
                market_cap REAL,
                pe_ratio REAL,
                dividend_yield REAL,
                debt_to_equity REAL,
                current_ratio REAL,
                return_on_equity REAL,
                profit_margin REAL,
                price_to_book REAL,
                beta REAL,
                fifty_day_avg REAL,
                two_hundred_day_avg REAL,
                market TEXT,
                last_updated TEXT
            )
        ''')
        self.conn.commit()

    # =======================
    # 3. Update Stock Prices
    # =======================
    # def update_stock_prices(self, force_update=True):
    #     """
    #     Update stock prices and key metrics in the database.
    #     Uses a new connection for thread-safety.
    #     Updates stocks not refreshed in the last 2 hours unless force_update is True.
    #     """
    #     # Open a new connection for this thread
    #     thread_conn = sqlite3.connect(self.db_path, check_same_thread=False)
    #     cursor = thread_conn.cursor()
        
    #     if force_update:
    #         cursor.execute("SELECT ticker, market FROM stocks")
    #     else:
    #         cursor.execute("""
    #             SELECT ticker, market FROM stocks 
    #             WHERE 
    #                 last_updated IS NULL OR 
    #                 (julianday('now') - julianday(last_updated)) * 24 > 2
    #         """)
    #     stocks_to_update = cursor.fetchall()
        
    #     import traceback
    #     for ticker in stocks_to_update:
    #         print(f"Attempting update for {ticker}")
    #         try:
    #             stock = yf.Ticker(ticker)
    #             info = stock.info
    #             print(info)
    #             if ticker.upper().startswith('BAJAJ'):
    #                 print(f"Raw info for {ticker}:{info}")
    #             update_query = """
    #                 UPDATE stocks SET 
    #                     current_price = ?,
    #                     market_cap = ?,
    #                     pe_ratio = ?,
    #                     dividend_yield = ?,
    #                     debt_to_equity = ?,
    #                     current_ratio = ?,
    #                     return_on_equity = ?,
    #                     profit_margin = ?,
    #                     price_to_book = ?,
    #                     beta = ?,
    #                     fifty_day_avg = ?,
    #                     two_hundred_day_avg = ?,
    #                     last_updated = ?
    #                 WHERE ticker = ?
    #             """
    #             update_values = (
    #                 info.get('currentPrice'),
    #                 info.get('marketCap'),
    #                 info.get('trailingPE'),
    #                 info.get('dividendYield', 0) * 100,  # Convert to percentage
    #                 info.get('debtToEquity'),
    #                 info.get('currentRatio'),
    #                 info.get('returnOnEquity'),
    #                 info.get('profitMargins', 0) * 100,    # Convert to percentage
    #                 info.get('priceToBook'),
    #                 info.get('beta'),
    #                 info.get('fiftyDayAverage'),
    #                 info.get('twoHundredDayAverage'),
    #                 datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    #                 ticker
    #             )
    #             cursor.execute(update_query, update_values)
    #             print(f"Updated {ticker} with price {info.get('currentPrice')}")
    #         except Exception as e:
    #             print(f"Error updating {ticker}: {e}")
    #             traceback.print_exc()
        
    #     thread_conn.commit()
    #     thread_conn.close()


    # def update_stock_prices_async(self, force_update=False):
    #     """Run the update_stock_prices method asynchronously."""
    #     thread = threading.Thread(target=self.update_stock_prices, args=(force_update,))
    #     thread.daemon = True  # Daemon thread won't block program exit
    #     thread.start()

    # =======================
    # 4. Financial Scoring and Recommendation Methods
    # =======================
    def calculate_financial_score(self, stock_data):
        """
        Calculate a financial health score based on various ratios.
        Returns a score between 0 and 100.
        """
        (ticker, company, sector, price, market_cap, pe_ratio, 
         dividend_yield, debt_to_equity, current_ratio, 
         return_on_equity, profit_margin, price_to_book, beta) = stock_data

        # Initialize scores
        pe_score = 0
        if pe_ratio and pe_ratio > 0:
            if 5 <= pe_ratio <= 25:
                pe_score = 20 - (abs(15 - pe_ratio) / 10) * 20
            else:
                pe_score = max(0, 10 - (abs(15 - pe_ratio) / 15) * 10)

        debt_score = 20 if debt_to_equity is not None and debt_to_equity <= 0.5 else (
                     15 if debt_to_equity <= 1 else (
                     10 if debt_to_equity <= 1.5 else (
                      5 if debt_to_equity <= 2 else 0)))
        
        if current_ratio is not None:
            if current_ratio >= 2:
                liquidity_score = 15
            elif current_ratio >= 1.5:
                liquidity_score = 12
            elif current_ratio >= 1:
                liquidity_score = 8
            else:
                liquidity_score = max(0, (current_ratio / 1) * 8)
        else:
            liquidity_score = 0

        # Profitability scoring using ROE and profit margin
        roe_score = 10 if return_on_equity and return_on_equity >= 15 else (
                    8 if return_on_equity >= 10 else (
                    5 if return_on_equity >= 5 else (
                    3 if return_on_equity > 0 else 0)))
        margin_score = 10 if profit_margin and profit_margin >= 20 else (
                       8 if profit_margin >= 10 else (
                       5 if profit_margin >= 5 else (
                       3 if profit_margin > 0 else 0)))
        profitability_score = roe_score + margin_score

        if price_to_book and price_to_book > 0:
            if price_to_book <= 1:
                valuation_score = 15
            elif price_to_book <= 3:
                valuation_score = 15 - ((price_to_book - 1) / 2) * 10
            else:
                valuation_score = max(0, 5 - ((price_to_book - 3) / 2) * 5)
        else:
            valuation_score = 0

        stability_score = max(0, 10 - abs(beta - 1) * 5) if beta is not None else 0

        dividend_bonus = min(5, dividend_yield) if dividend_yield is not None else 0

        total_score = pe_score + debt_score + liquidity_score + profitability_score + valuation_score + stability_score + dividend_bonus
        return min(100, total_score)

    def get_recommendations(self, budget, market='US', currency='USD', sector=None, criteria=None, limit=5):
        """
        Return stock recommendations based on the provided budget and optional filters.
        Triggers asynchronous updates so the response isn't delayed.
        """
        # Start background update (won't block main thread)
        # self.update_stock_prices_async(force_update=False)
        
        cursor = self.conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM stocks WHERE market = ?", (market,))
        count = cursor.fetchone()[0]
        if count == 0:
            return {"message": f"No {market} stocks in database. Please update database first."}

        # Build the query
        query = """
            SELECT ticker, company_name, sector, current_price, market_cap, pe_ratio, 
                   dividend_yield, debt_to_equity, current_ratio, return_on_equity, 
                   profit_margin, price_to_book, beta 
            FROM stocks 
            WHERE current_price > 0 AND market = ?
        """
        params = [market]
        if sector:
            query += " AND sector = ?"
            params.append(sector)
        if criteria:
            if 'sectors' in criteria and not sector:
                query += " AND sector IN ({})".format(','.join('?' * len(criteria['sectors'])))
                params.extend(criteria['sectors'])
            if 'min_price' in criteria:
                query += " AND current_price >= ?"
                params.append(criteria['min_price'])
            if 'max_price' in criteria:
                query += " AND current_price <= ?"
                params.append(criteria['max_price'])
            if 'min_dividend' in criteria:
                query += " AND dividend_yield >= ?"
                params.append(criteria['min_dividend'])
            if 'max_pe' in criteria:
                query += " AND pe_ratio <= ?"
                params.append(criteria['max_pe'])
            if 'max_debt_to_equity' in criteria:
                query += " AND debt_to_equity <= ?"
                params.append(criteria['max_debt_to_equity'])
            if 'min_current_ratio' in criteria:
                query += " AND current_ratio >= ?"
                params.append(criteria['min_current_ratio'])
            if 'min_roe' in criteria:
                query += " AND return_on_equity >= ?"
                params.append(criteria['min_roe'])
                
        cursor.execute(query, params)
        results = cursor.fetchall()
        if not results:
            return {"message": "No stocks found based on given criteria. Try adjusting your filters."}

        # Compute financial and value scores
        scored_stocks = []
        for stock in results:
            fin_score = self.calculate_financial_score(stock)
            price = stock[3]
            max_shares = int(budget / price)
            if max_shares > 0:
                value_score = fin_score * (max_shares ** 0.5)
                scored_stocks.append((stock, fin_score, value_score, max_shares))
        
        if not scored_stocks:
            cheapest = min(results, key=lambda x: x[3])
            currency_symbol = '₹' if currency == 'INR' else '$'
            return {"message": f"Your budget of {currency_symbol}{round(budget,2)} is too low for available stocks. The cheapest is {cheapest[1]} ({cheapest[0]}) at {currency_symbol}{round(cheapest[3],2)} per share."}

        # Sort stocks by value score and ensure sector diversity
        scored_stocks.sort(key=lambda x: x[2], reverse=True)
        selected_stocks = []
        sectors_selected = set()
        for stock, fin_score, value_score, max_shares in scored_stocks:
            if stock[2] not in sectors_selected and len(selected_stocks) < limit:
                selected_stocks.append((stock, fin_score, value_score, max_shares))
                sectors_selected.add(stock[2])
        if len(selected_stocks) < limit:
            remaining = [s for s in scored_stocks if s not in selected_stocks]
            selected_stocks.extend(remaining[:limit - len(selected_stocks)])
        selected_stocks.sort(key=lambda x: x[2], reverse=True)

        # Build the recommendation list
        recommendations = []
        remaining_budget = budget
        for stock, fin_score, value_score, max_shares in selected_stocks:
            ticker, company, sector, price = stock[:4]
            if max_shares >= 1 and remaining_budget >= price:
                relative_value = value_score / sum(s[2] for s in selected_stocks)
                target_allocation = relative_value * budget
                max_allocation = remaining_budget * 0.4
                allocation = min(target_allocation, max_allocation)
                shares = max(1, min(max_shares, int(allocation / price)))
                cost = shares * price
                remaining_budget -= cost

                financial_ratios = {
                    "PE Ratio": f"{stock[5]:.2f}" if stock[5] else "N/A",
                    "Dividend Yield": f"{stock[6]:.2f}%" if stock[6] else "N/A",
                    "Debt to Equity": f"{stock[7]:.2f}" if stock[7] else "N/A",
                    "Current Ratio": f"{stock[8]:.2f}" if stock[8] else "N/A",
                    "Return on Equity": f"{stock[9]:.2f}%" if stock[9] else "N/A",
                    "Profit Margin": f"{stock[10]:.2f}%" if stock[10] else "N/A",
                    "Price to Book": f"{stock[11]:.2f}" if stock[11] else "N/A",
                    "Beta": f"{stock[12]:.2f}" if stock[12] else "N/A",
                    "Financial Health Score": f"{fin_score:.1f}/100",
                    "Value Score": f"{value_score:.1f}"
                }
                recommendations.append({
                    "ticker": ticker,
                    "company": company,
                    "sector": sector,
                    "price_per_share": price,
                    "shares": shares,
                    "total_cost": cost,
                    "allocation_percent": (cost / budget) * 100,
                    "financial_score": fin_score,
                    "financial_ratios": financial_ratios
                })
                # Stop if budget is nearly exhausted
                if remaining_budget < min([s[0][3] for s in scored_stocks if s[0][3] <= remaining_budget], default=float('inf')):
                    break

        currency_symbol = '₹' if currency == 'INR' else '$'
        return {
            "budget": budget,
            "currency": currency_symbol,
            "spent": budget - remaining_budget,
            "remaining": remaining_budget,
            "recommendations": recommendations
        }

    # =======================
    # 5. Explanation Methods and Cleanup
    # =======================
    def explain_recommendation(self, ticker):
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT ticker, company_name, sector, current_price, market_cap, 
                   pe_ratio, dividend_yield, debt_to_equity, current_ratio, 
                   return_on_equity, profit_margin, price_to_book, beta, market
            FROM stocks WHERE ticker = ?
        """, (ticker,))
        stock = cursor.fetchone()
        if not stock:
            return {"message": f"Stock {ticker} not found in database."}
        fin_score = self.calculate_financial_score(stock[:13])
        strengths, weaknesses = [], []
        if stock[5]:
            if 10 <= stock[5] <= 20:
                strengths.append("Favorable P/E ratio")
            elif stock[5] > 30:
                weaknesses.append("High P/E ratio")
        summary = self.generate_recommendation_summary(stock[:13], fin_score, strengths, weaknesses, stock[13])
        return {
            "ticker": stock[0],
            "company": stock[1],
            "sector": stock[2],
            "current_price": stock[3],
            "market_cap": stock[4],
            "pe_ratio": stock[5],
            "dividend_yield": stock[6],
            "debt_to_equity": stock[7],
            "current_ratio": stock[8],
            "return_on_equity": stock[9],
            "profit_margin": stock[10],
            "price_to_book": stock[11],
            "beta": stock[12],
            "market": "Indian" if stock[13] == "IN" else "US",
            "financial_score": f"{fin_score:.1f}/100",
            "strengths": strengths,
            "weaknesses": weaknesses,
            "summary": summary
        }

    def generate_recommendation_summary(self, stock, score, strengths, weaknesses, market):
        ticker, company, sector = stock[0], stock[1], stock[2]
        currency_symbol = "₹" if market == "IN" else "$"
        if score >= 75:
            rating = "Strong Buy"
            reason = "excellent financial health and attractive valuation metrics"
        elif score >= 60:
            rating = "Buy"
            reason = "good overall financial indicators"
        elif score >= 45:
            rating = "Hold"
            reason = "mixed financial indicators"
        else:
            rating = "Not Recommended"
            reason = "concerning financial metrics"
        market_text = "Indian" if market == "IN" else "US"
        summary = f"{company} ({ticker}) from the {market_text} market is rated as a {rating} based on {reason}."
        if strengths:
            summary += f" Key strengths include {strengths[0].lower()}."
            if len(strengths) > 1:
                summary += f" and {strengths[1].lower()}."
        if weaknesses and rating != "Strong Buy":
            summary += f" However, investors should consider {weaknesses[0].lower()}."
            if len(weaknesses) > 1:
                summary += f" and {weaknesses[1].lower()}."
        return summary

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None
# =======================
# 6. LangChain Tools / Utility Functions
# =======================
def setup_database_tool():
    engine = StockRecommendationEngine()
    engine.close()
    return "Database setup complete."

def get_recommendations_tool(budget, market='US', currency='USD', sector=None, criteria=None, limit=5):
    engine = StockRecommendationEngine()
    recs = engine.get_recommendations(budget, market, currency, sector, criteria, limit)
    engine.close()
    return recs

def explain_recommendation_tool(ticker):
    engine = StockRecommendationEngine()
    explanation = engine.explain_recommendation(ticker)
    engine.close()
    return explanation


# import json 

# a = get_recommendations_tool(1000, market='US', currency='USD', sector='Technology', criteria=None, limit=5)

# print(json.dumps(a, indent=4))
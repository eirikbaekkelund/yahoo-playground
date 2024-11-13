import pandas as pd

def analyst_ratios(df: pd.DataFrame) -> pd.DataFrame:
    
    strong_buy = df["strongBuy"]
    buy = df["buy"]
    sell = df["sell"]
    strong_sell = df["strongSell"]

    sentiment_score = ((strong_buy * 2 + buy) - (sell + strong_sell * 2)) / (strong_buy + buy + sell + strong_sell)
    buy_sell_ratio = (strong_buy + buy + 1) / (sell + strong_sell + 1)

    df["sentiment_score"] = sentiment_score
    df["buy_sell_ratio"] = buy_sell_ratio

    return df

def rolling_ratios(df: pd.DataFrame) -> pd.DataFrame:
    df["price_to_monthly_roll"] = df["price"] / df["close_monthly_roll"]
    df["price_to_quarterly_roll"] = df["price"] / df["close_quarterly_roll"]

    return df

def weighted_price(df: pd.DataFrame) -> pd.DataFrame:
    df["weighted_price_rec"] = df["sentiment_score"] * df["price"]

    return df

def balance_sheet_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df['debt_to_asset_ratio'] = df['Total Debt'] / df['Total Assets']
    df['working_capital'] = df['Current Assets'] - df['Current Liabilities']
    df['book_val_per_share'] = df['Tangible Book Value'] / df['Ordinary Shares Number']
    df['net_debt'] = df['Total Debt'] - df['Cash And Cash Equivalents']
    df['earnings_to_asset_ratio'] = df['Retained Earnings'] / df['Total Assets']
    df['earnings'] = df['Retained Earnings']

    return df

def beta(stock: pd.DataFrame, market: pd.DataFrame, column: str) -> float:
    """ 
    beta = cov(returns, market_returns) / var(market_returns)
    """
    #TODO: beta at monthly levels
    stock_pct = stock[column].pct_change()
    sp500_pct = market[column].pct_change()
    beta = stock_pct.cov(sp500_pct) / sp500_pct.var()

    return beta

def moving_average(df: pd.DataFrame, column: str, window: int) -> pd.Series:
    # TODO: get mean and std dev at monthly level
    data = df[column].rolling(window=window).mean()
    # fill missing values with the actual value
    data.fillna(df[column], inplace=True)
    
    return data.dropna()

def pct_change_signals(df: pd.DataFrame) -> pd.DataFrame:
    df["pct_change"] = df["Close"].pct_change().fillna(0)
    df["signal"] = df["pct_change"].apply(lambda x: 1 if x > 0.05 else -1 if x < -0.05 else 0)

    return df

def change_in_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    recommendations = ["strongBuy", "buy", "hold", "sell", "strongSell"]
    ticker_recs = df.groupby("date")[recommendations].sum()
    ticker_recs = ticker_recs.cumsum()
    change_in_recs = ticker_recs.diff().fillna(0)
    df = df.merge(change_in_recs, on="date", how="left", suffixes=("", "_change"))
        
    return df
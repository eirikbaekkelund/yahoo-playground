import asyncio
import requests
import yfinance as yf
import pandas as pd
import datetime
from data.config import TickerData
from data.constants import (
    NASDAQ_URL,
    SP500_URL,
    HISTORY_METRICS,
    BALANCE_SHEET_METRICS,
)
from data.feature_engineering import (
    balance_sheet_metrics,
    moving_average,
    analyst_ratios,
)
from tqdm.asyncio import tqdm
from typing import List

def get_valid_tickers(tickers: List[str]) -> List[str]:
    return [ticker for ticker in tickers if ticker.isalpha()]

def get_nasdaq_tickers() -> List[str]:
    resp = requests.get(NASDAQ_URL)

    data = resp.text.split('\n')
    data = [row.split('|') for row in data]
    df = pd.DataFrame(data[1:], columns=data[0])

    df.columns = df.columns.str.replace('\r', '', regex=False)
    df = df.map(lambda x: x.replace('\r', '') if isinstance(x, str) else x)

    df['Compall Name'] = df['Security Name'].str.split(' - ').str[0]
    last_col = df.pop(df.columns[-1])
    df.insert(1, last_col.name, last_col)
    
    tickers =  df['Symbol'].tolist()
    
    return get_valid_tickers(tickers)

def get_sp500_tickers() -> List[str]:
    sp500_df = pd.read_html(SP500_URL)[0]
    tickers =  sp500_df['Symbol'].tolist()

    return get_valid_tickers(tickers)

  
def get_timeseries(yf_ticker: yf.Ticker, start: str, end: str) -> dict:
    df = yf_ticker.history(start=start, end=end)

    if not all([m in df.columns for m in HISTORY_METRICS]):
        return None
    
    df = df[HISTORY_METRICS]
    
    close_monthly_roll = moving_average(df, 'Close', 30)
    close_quarterly_roll = moving_average(df, 'Close', 90)
    df["price_monthly_roll"] = close_monthly_roll
    df["price_quarterly_roll"] = close_quarterly_roll
    # set index to yyyy-mm-dd
    df.index = pd.to_datetime(df.index.date)
    
    return df

def align_dataframes(df: pd.DataFrame, ref: pd.DataFrame, fill_method: str = 'ffil') -> pd.DataFrame:
    df.index = pd.to_datetime(df.index)
    seen_dates = set()
    df["date"] = df.index   
    for date in df.index:
        closest_date = min(ref.index, key=lambda x: abs(x - date))
        if pd.Timedelta(closest_date - date).days > 90:
            continue
        elif closest_date != date and closest_date not in df.index and closest_date not in seen_dates:
            df.at[date, "date"] = closest_date
            seen_dates.add(closest_date)
    df.set_index("date", inplace=True)
    
    return pd.concat([df, ref], axis=1, join='inner')

def get_analyst_recs(yf_ticker: yf.Ticker) -> dict:
    try:
        recommendations = yf_ticker.recommendations
        if recommendations.empty:
            return {}
        df = analyst_ratios(recommendations)
        month_offsets = df["period"].str.extract(r"(\d+)([mM])")
        month_offsets = month_offsets.rename(columns={0: "offset", 1: "unit"})
        df["date"] = month_offsets.apply(lambda x: pd.Timestamp.today() - pd.DateOffset(months=int(x["offset"])), axis=1).dt.date
        df = df.drop(columns=["period"])
        df.set_index('date', inplace=True)
        
        return df
    # except 404 client error
    except Exception as e:
        return None
    

def get_balancesheet_metrics(df: pd.DataFrame) -> dict:
    df = df.T
    if not all([m in df.columns for m in BALANCE_SHEET_METRICS]):
        return None
    df = df[BALANCE_SHEET_METRICS]
    df = balance_sheet_metrics(df)
    
    return df

async def fetch_ticker_data(ticker: str,  start: str, end: str) -> TickerData | None:
    try:
        yf_ticker = yf.Ticker(ticker)
        info = yf_ticker.info
        if not info:
            return None
        
        prices = get_timeseries(yf_ticker, start, end)
        analyst_recs = get_analyst_recs(yf_ticker)
        balance_sheet = get_balancesheet_metrics(yf_ticker.quarterly_balance_sheet)
        dfs = [prices, analyst_recs, balance_sheet]
        if any([not isinstance(x, pd.DataFrame) for x in dfs]):
            return None
        
        aligned_balance_sheet = align_dataframes(balance_sheet, prices)
        aligned_analyst_recs = align_dataframes(analyst_recs, prices)
        
        dfs = [prices, aligned_analyst_recs, aligned_balance_sheet]
        for df in dfs:
            df["Close"] = df["Close"] + df["Dividends"]
            df.drop(columns=["Dividends"], inplace=True)
            df["ticker"] = ticker

    except Exception as e:
        # Handle errors, return None or some indication of failure
        print(f"Error fetching data for {ticker}:\n{e}")
        return None
    
    return TickerData(
        ticker=ticker,
        prices=prices,
        analyst_recs=aligned_analyst_recs,
        balance_sheet=aligned_balance_sheet,
    )


async def get_ticker_data(tickers: List[str], start: str, end: str) -> List[TickerData]:
    tasks = [fetch_ticker_data(ticker, start, end) for ticker in tickers]
    results = await tqdm.gather(*tasks)
    
    return [r for r in results if r is not None]

async def batch_download_data(tickers: List[str], batch_size: int, start: str, end: str) -> List[TickerData]:
    batches =  [tickers[i:i + batch_size] for i in range(0, len(tickers), batch_size)]
    ticker_data = []
    for i, batch in enumerate(batches):
        print(f"Fetching batch {i+1}/{len(batches)}", end='\r')
        batch_data = await get_ticker_data(batch, start, end)
        ticker_data.extend(batch_data)
    
    return ticker_data


async def collect_sp500_data(lookback_days: int = 365*2, batch_size: int = 2000) -> List[TickerData]:
    end = datetime.datetime.now().strftime('%Y-%m-%d')
    start = (datetime.datetime.now() - datetime.timedelta(days=lookback_days)).strftime('%Y-%m-%d')
    sp500_tickers = get_sp500_tickers()

    return await batch_download_data(sp500_tickers, batch_size, start, end)


if __name__ == "__main__":
    import asyncio
    data = asyncio.run(collect_sp500_data())
    
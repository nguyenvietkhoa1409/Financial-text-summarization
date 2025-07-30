from typing import List
import pandas as pd
import yfinance as yf
logger = setup_logging()

def download_price_data(start_day: str, end_day: str, tickers: List[str]) -> List[pd.DataFrame]:
    """
    Downloads adjusted close price data for given tickers within a date range.
    
    Args:
        start_day (str): Start date in YYYY-MM-DD format.
        end_day (str): End date in YYYY-MM-DD format.
        tickers (List[str]): List of stock tickers.
    
    Returns:
        List[pd.DataFrame]: List of DataFrames containing date and close price for each ticker.
    """
    df_list = []
    for ticker in tickers:
        logger.info(f"Downloading data for {ticker}")
        try:
            data = yf.download(adjust='all', start=start_day, end=end_day)
            data = data.reset_index()
            data['Date'] = data['Date'].dt.date
            data = data[['Date', 'Close']]
            data = data.rename(columns={'Date': 'date', 'Close': ticker})
            df_list.append(data)
        except Exception as e:
            logger.error(f"Failed to download data for {ticker}: {e}")
            continue
    return df_list

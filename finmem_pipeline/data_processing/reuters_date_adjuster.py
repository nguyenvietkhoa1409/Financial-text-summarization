from datetime import timedelta
from tqdm import tqdm
logger = setup_logging()

def build_trading_date_map(trading_dates, start_date, end_date):
    """
    Builds a mapping from each calendar date to the next trading day.
    
    Args:
        trading_dates (list or pd.Series): List of valid trading dates.
        start_date (pd.Timestamp): Start of calendar date range.
        end_date (pd.Timestamp): End of calendar date range.
    
    Returns:
        dict: Mapping from calendar dates to trading dates.
    """
    try:
        trading_dates = pd.to_datetime(sorted(pd.to_datetime(trading_dates).normalize()))
        calendar_dates = pd.date_range(start=start_date, end=end_date, freq='D')
        mapping = {}
        trading_idx = 0
        for date in tqdm(calendar_dates, desc="Building trading date map"):
            while trading_idx < len(trading_dates) and trading_dates[trading_idx] < date:
                trading_idx += 1
            if trading_idx < len(trading_dates):
                mapping[date] = trading_dates[trading_idx]
            else:
                mapping[date] = trading_dates[-1]
        return mapping
    except Exception as e:
        logger.error(f"Failed to build trading date map: {e}")
        raise

def adjust_trading_days_fast(df: pd.DataFrame, trading_date_map: dict) -> pd.DataFrame:
    """
    Adjusts dates in a DataFrame to the next trading day using a precomputed map.
    
    Args:
        df (pd.DataFrame): DataFrame with a 'date' column.
        trading_date_map (dict): Mapping from raw dates to trading dates.
    
    Returns:
        pd.DataFrame: Updated DataFrame with adjusted 'date' column.
    """
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date']).dt.normalize()
        tqdm.pandas(desc="Adjusting to trading days")
        df['date'] = df['date'].progress_map(lambda x: trading_date_map.get(x, x))
        return df
    except Exception as e:
        logger.error(f"Failed to adjust trading days: {e}")
        raise

def process_reuters_dates(news_data: pd.DataFrame, trading_dates: list, output_path: str) -> pd.DataFrame:
    """
    Process Reuters news dates to align with trading days.
    
    Args:
        news_data (pd.DataFrame): News DataFrame with 'date' column.
        trading_dates (list): List of trading dates.
        output_path (str): Path to save the processed DataFrame.
    
    Returns:
        pd.DataFrame: Processed DataFrame with adjusted dates.
    """
    try:
        news_data['date'] = pd.to_datetime(news_data['date']).dt.normalize()
        start_date = news_data['date'].min()
        end_date = news_data['date'].max()

        trading_date_map = build_trading_date_map(trading_dates, start_date, end_date)
        
        news_raw = news_data[~news_data['date'].isin(trading_dates)]
        news_processed = news_data[news_data['date'].isin(trading_dates)]
        
        df_adjusted = adjust_trading_days_fast(news_raw, trading_date_map)
        
        df_clean = pd.concat([news_processed, df_adjusted]).sort_values(by='date').reset_index(drop=True)
        
        df_clean.to_parquet(output_path, index=False)
        logger.info(f"Processed news data saved to: {output_path}")
        
        return df_clean
    except Exception as e:
        logger.error(f"Failed to process Reuters dates: {e}")
        raise

rom typing import List
import pandas as pd
logger = setup_logging()

def combine_price_dataframes(df_list: List[pd.DataFrame], path: str, tickers: List[str]) -> dict:
    """
    Combines dataframes of different tickers into a single dictionary and saves it as a pickle file.
    
    Args:
        df_list (List[pd.DataFrame]): List of price DataFrames.
        path (str): Directory path to save the pickle file.
        tickers (List[str]): List of stock tickers.
    
    Returns:
        dict: Combined price dictionary.
    """
    try:
        df_dicts = [dict(zip(df['date'], df[ticker])) for df, ticker in zip(df_list, tickers)]
        combined_dict = {date: {'price': {}} for df_dict in df_dicts for date in df_dict}
        for i, df_dict in enumerate(df_dicts):
            for date, price in df_dict.items():
                combined_dict[date]['price'][tickers[i]] = price
        pkl_filename = f"{path}price.pkl"
        save_pickle(combined_dict, pkl_filename)
        return combined_dict
    except Exception as e:
        logger.error(f"Failed to combine price dataframes: {e}")
        raise
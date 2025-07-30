import polars as pl
import os
logger = setup_logging()

def combine_news_data(news_df: pl.DataFrame, reuters_df: pl.DataFrame, output_path: str) -> pl.DataFrame:
    """
    Combines existing news data with Reuters news data and saves to parquet.
    
    Args:
        news_df (pl.DataFrame): Existing news DataFrame.
        reuters_df (pl.DataFrame): Reuters news DataFrame.
        output_path (str): Path to save the combined parquet file.
    
    Returns:
        pl.DataFrame: Combined and sorted news DataFrame.
    """
    try:
        reuters_df = reuters_df.with_columns([
            pl.lit(None, dtype=pl.String).alias("author"),
            pl.lit(None, dtype=pl.String).alias("source"),
            pl.lit(None, dtype=pl.String).alias("summary"),
            pl.lit(None, dtype=pl.String).alias("url")
        ])
        reuters_df = reuters_df.with_columns([
            pl.col("datetime").cast(pl.Datetime(time_unit='us')).alias("datetime"),
            pl.col("date").cast(pl.Datetime(time_unit='us')).alias("date")
        ])
        
        desired_columns = ['author', 'content', 'datetime', 'source', 'summary', 'title', 'url', 'date', 'equity']
        reuters_df = reuters_df.select(desired_columns)
        
        combined_df = pl.concat([news_df, reuters_df])
        sorted_combined_df = combined_df.sort("date", descending=False)
        
        save_parquet(sorted_combined_df, output_path)
        
        return sorted_combined_df
    except Exception as e:
        logger.error(f"Failed to combine news data: {e}")
        raise

def load_news_data(path: str) -> pl.DataFrame:
    """
    Load news data from a parquet file.
    
    Args:
        path (str): Path to the parquet file.
    
    Returns:
        pl.DataFrame: Loaded news DataFrame.
    """
    if os.path.exists(path):
        return read_parquet(path)
    else:
        logger.error(f"News file not found at: {path}")
        raise FileNotFoundError(f"News file not found at: {path}")
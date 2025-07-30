import os
import pickle
import pandas as pd
import polars as pl
from pathlib import Path
from finmem_pipeline.utils.logging import setup_logging

logger = setup_logging()

def save_pickle(data, path: str):
    """
    Save data as a pickle file.

    Args:
        data: Data to save.
        path (str): File path.
    """
    try:
        with open(path, 'wb') as file:
            pickle.dump(data, file)
        logger.info(f"Data saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to save pickle file {path}: {e}")
        raise

def read_pickle(path: str):
    """
    Read data from a pickle file.

    Args:
        path (str): File path.

    Returns:
        Data loaded from the pickle file.
    """
    try:
        with open(path, 'rb') as file:
            data = pickle.load(file)
        logger.info(f"Data loaded from: {path}")
        return data
    except Exception as e:
        logger.error(f"Failed to read pickle file {path}: {e}")
        raise

def save_parquet(df: pl.DataFrame, path: str):
    """
    Save Polars DataFrame as a parquet file.

    Args:
        df (pl.DataFrame): DataFrame to save.
        path (str): File path.
    """
    try:
        df.write_parquet(path)
        logger.info(f"Parquet file saved to: {path}")
    except Exception as e:
        logger.error(f"Failed to save parquet file {path}: {e}")
        raise

def read_parquet(path: str) -> pl.DataFrame:
    """
    Read data from a parquet file.

    Args:
        path (str): File path.

    Returns:
        pl.DataFrame: DataFrame loaded from the parquet file.
    """
    try:
        df = pl.read_parquet(path)
        logger.info(f"Parquet file loaded from: {path}")
        return df
    except Exception as e:
        logger.error(f"Failed to read parquet file {path}: {e}")
        raise
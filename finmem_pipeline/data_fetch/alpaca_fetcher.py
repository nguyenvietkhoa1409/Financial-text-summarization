import os
import time
import shutil
import httpx
import tenacity
import polars as pl
import pandas as pd
from datetime import date, timedelta, datetime
from typing import List, Tuple, Union
from tenacity import retry, stop_after_attempt, wait_fixed
from alpaca.data import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from finmem_pipeline.utils.logging import setup_logging
from finmem_pipeline.utils.storage import read_pickle, save_parquet

logger = setup_logging()

END_POINT_TEMPLATE = "https://data.alpaca.markets/v1beta1/news?start={start_date}&end={end_date}&limit=50&symbols={symbol}"
END_POINT_TEMPLATE_LINK_PAGE = "https://data.alpaca.markets/v1beta1/news?limit=50&symbol={symbol}&page_token={page_token}"
NUM_NEWS_PER_RECORD = 200
MAX_ATTEMPTS = 5
WAIT_TIME = 60

def round_to_next_day(date: pl.Expr) -> pl.Expr:
    """
    Rounds datetime to the next day if the time is after 4:00 PM.

    Args:
        date (pl.Expr): Polars expression for datetime column.

    Returns:
        pl.Expr: Polars expression for adjusted date.
    """
    condition = (date.dt.hour() >= 16) & ((date.dt.minute() > 0) | (date.dt.second() > 0))
    return pl.when(condition).then(date.dt.offset_by("1d")).otherwise(date)

class ScraperError(Exception):
    """Custom exception for Alpaca API scraping errors."""
    pass

class RecordContainerFull(Exception):
    """Custom exception raised when record container is full."""
    pass

class ParseRecordContainer:
    def __init__(self, symbol: str) -> None:
        """
        Initialize a container to store news records for a symbol.

        Args:
            symbol (str): Stock ticker symbol.
        """
        self.symbol = symbol
        self.record_counter = 0
        self.author_list = []
        self.content_list = []
        self.date_list = []
        self.source_list = []
        self.summary_list = []
        self.title_list = []
        self.url_list = []

    def add_records(self, records: List[dict]) -> None:
        """
        Add news records to the container.

        Args:
            records (List[dict]): List of news records.

        Raises:
            RecordContainerFull: If the container reaches NUM_NEWS_PER_RECORD.
        """
        for cur_record in records:
            self.author_list.append(cur_record.get("author"))
            self.content_list.append(cur_record.get("content"))
            date_str = cur_record.get("created_at", "").rstrip("Z")
            self.date_list.append(datetime.fromisoformat(date_str) if date_str else None)
            self.source_list.append(cur_record.get("source"))
            self.summary_list.append(cur_record.get("summary"))
            self.title_list.append(cur_record.get("headline"))
            self.url_list.append(cur_record.get("url"))
            self.record_counter += 1
            if self.record_counter == NUM_NEWS_PER_RECORD:
                raise RecordContainerFull

    def pop(self, align_next_date: bool = True) -> Union[pl.DataFrame, None]:
        """
        Create a Polars DataFrame from collected records.

        Args:
            align_next_date (bool): Whether to align dates to the next trading day.

        Returns:
            pl.DataFrame or None: DataFrame of news records or None if empty.
        """
        if self.record_counter == 0:
            return None
        return_df = pl.DataFrame(
            {
                "author": self.author_list,
                "content": self.content_list,
                "datetime": self.date_list,
                "source": self.source_list,
                "summary": self.summary_list,
                "title": self.title_list,
                "url": self.url_list,
            }
        )
        if align_next_date:
            return_df = return_df.with_columns(
                round_to_next_day(return_df["datetime"]).alias("date"),
            )
        else:
            return_df = return_df.with_columns(
                pl.col("datetime").dt.date().alias("date"),
            )
        return return_df.with_columns(pl.lit(self.symbol).alias("equity"))

@retry(stop=stop_after_attempt(MAX_ATTEMPTS), wait=wait_fixed(WAIT_TIME))
def query_one_record(args: Tuple[date, str], temp_dir: str, api_key: str, secret_key: str, include_content: bool = False, exclude_contentless: bool = False) -> None:
    """
    Query news data for a single date and symbol, saving to a parquet file.

    Args:
        args (Tuple[date, str]): Tuple of date and symbol.
        temp_dir (str): Temporary directory for storing parquet files.
        api_key (str): Alpaca API key.
        secret_key (str): Alpaca secret key.
        include_content (bool): Whether to include content in the request.
        exclude_contentless (bool): Whether to exclude news without content.

    Raises:
        ScraperError: If the API request fails.
    """
    date, symbol = args
    next_date = date + timedelta(days=1)
    request_header = {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
    }
    container = ParseRecordContainer(symbol)
    with httpx.Client() as client:
        url = END_POINT_TEMPLATE.format(
            start_date=date.strftime("%Y-%m-%d"),
            end_date=next_date.strftime("%Y-%m-%d"),
            symbol=symbol,
        )
        if include_content:
            url += "&include_content=True"
        if exclude_contentless:
            url += "&exclude_contentless=True"

        response = client.get(url, headers=request_header)
        if response.status_code != 200:
            logger.error(f"API request failed for {symbol} on {date}: {response.text}")
            raise ScraperError(response.text)
        result = response.json()
        next_page_token = result.get("next_page_token")
        container.add_records(result.get("news", []))

        while next_page_token:
            try:
                url = END_POINT_TEMPLATE_LINK_PAGE.format(symbol=symbol, page_token=next_page_token)
                if include_content:
                    url += "&include_content=True"
                if exclude_contentless:
                    url += "&exclude_contentless=True"

                response = client.get(url, headers=request_header)
                if response.status_code != 200:
                    raise ScraperError(response.text)
                result = response.json()
                next_page_token = result.get("next_page_token")
                container.add_records(result.get("news", []))
            except RecordContainerFull:
                break

    result = container.pop(align_next_date=True)
    if result is not None:
        save_parquet(result, os.path.join(temp_dir, f"{uuid4()}.parquet"))
        logger.info(f"Saved news data for {symbol} on {date} to parquet")

def fetch_alpaca_stock_data(api_key: str, secret_key: str, start_date: str, end_date: str, tickers: list) -> pl.DataFrame:
    """
    Fetch stock data from Alpaca API.

    Args:
        api_key (str): Alpaca API key.
        secret_key (str): Alpaca secret key.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        tickers (list): List of stock tickers.

    Returns:
        pl.DataFrame: Polars DataFrame containing stock data.
    """
    try:
        client = StockHistoricalDataClient(api_key, secret_key)
        request_params = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )
        bars = client.get_stock_bars(request_params)
        df = pl.from_pandas(bars.df.reset_index())
        logger.info(f"Fetched stock data for {tickers} from Alpaca API")
        return df
    except Exception as e:
        logger.error(f"Failed to fetch Alpaca stock data: {e}")
        raise

def fetch_alpaca_news_data(api_key: str, secret_key: str, start_date: str, end_date: str, tickers: list) -> pl.DataFrame:
    """
    Fetch news data from Alpaca API and save to parquet.

    Args:
        api_key (str): Alpaca API key.
        secret_key (str): Alpaca secret key.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.
        tickers (list): List of stock tickers.

    Returns:
        pl.DataFrame: Polars DataFrame containing news data.
    """
    try:
        from finmem_pipeline.config import load_config
        config = load_config()
        news_data_base_path = config['data_paths']['news_data']
        temp_dir = os.path.join(news_data_base_path, "temp")
        primary_dir = os.path.join(news_data_base_path, "03_primary")
        price_data_path = os.path.join(config['data_paths']['price_data'], "price.pkl")

        # Prepare directories
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        os.makedirs(temp_dir, exist_ok=True)
        os.makedirs(primary_dir, exist_ok=True)

        # Load price data to get unique (date, equity) pairs
        combined_price_data = read_pickle(price_data_path)
        price_records = []
        for date_key, data in combined_price_data.items():
            if 'price' in data:
                for ticker, price in data['price'].items():
                    price_records.append({"date": date_key, "equity": ticker, "price": price})
        data = pl.DataFrame(price_records)

        # Extract unique (date, equity) pairs
        query_data = data.select(["date", "equity"]).unique().to_dict()
        args_list = list(zip(query_data["date"], query_data["equity"]))

        # Fetch news data for each (date, equity) pair
        from tqdm import tqdm
        with tqdm(total=len(args_list)) as pbar:
            for i, arg in enumerate(args_list):
                try:
                    query_one_record(arg, temp_dir, api_key, secret_key, include_content=True, exclude_contentless=True)
                except tenacity.RetryError as e:
                    logger.error(f"Retry error for {arg}: {e}")
                pbar.update(1)
                if (i + 1) % 3000 == 0:
                    time.sleep(90)

        # Combine temporary files
        record_dfs = [
            pl.read_parquet(os.path.join(temp_dir, f))
            for f in os.listdir(temp_dir) if f.endswith(".parquet")
        ]

        if record_dfs:
            df = pl.concat(record_dfs)
            output_path = os.path.join(primary_dir, "news.parquet")
            save_parquet(df, output_path)
            logger.info(f"Combined news data saved to: {output_path}, shape: {df.shape}")
            return df
        else:
            logger.warning("No news data was successfully downloaded and processed.")
            return pl.DataFrame()

    except Exception as e:
        logger.error(f"Failed to fetch Alpaca news data: {e}")
        raise
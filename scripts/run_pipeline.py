import pandas as pd
from finmem_pipeline.config import load_config, ensure_directories
from finmem_pipeline.utils.logging import setup_logging
from finmem_pipeline.data_fetch.yfinance_fetcher import download_price_data
from finmem_pipeline.data_fetch.alpaca_fetcher import fetch_alpaca_stock_data, fetch_alpaca_news_data
from finmem_pipeline.data_fetch.sec_fetcher import fetch_sec_data
from finmem_pipeline.data_fetch.reuters_crawler import crawl_reuters_news
from finmem_pipeline.data_processing.price_processor import combine_price_dataframes
from finmem_pipeline.data_processing.news_processor import combine_news_data, load_news_data
from finmem_pipeline.data_processing.reuters_date_adjuster import process_reuters_dates

logger = setup_logging()

def main():
    """
    Main function to run the FinMem data pipeline.
    """
    try:
        config = load_config()
        ensure_directories(config)
        logger.info("Configuration loaded and directories ensured")

        price_df_list = download_price_data(
            config['fetch_params']['start_date'],
            config['fetch_params']['end_date'],
            config['fetch_params']['tickers']
        )
        logger.info("Price data fetched from YFinance")

        combine_price_dataframes(
            price_df_list,
            config['data_paths']['price_data'],
            config['fetch_params']['tickers']
        )
        logger.info("Price data combined and saved")

        alpaca_stock_df = fetch_alpaca_stock_data(
            config['alpaca_api_key'],
            config['alpaca_secret_key'],
            config['fetch_params']['start_date'],
            config['fetch_params']['end_date'],
            config['fetch_params']['tickers']
        )
        alpaca_news_df = fetch_alpaca_news_data(
            config['alpaca_api_key'],
            config['alpaca_secret_key'],
            config['fetch_params']['start_date'],
            config['fetch_params']['end_date'],
            config['fetch_params']['tickers']
        )
        logger.info("Alpaca stock and news data fetched")

        sec_data = fetch_sec_data(
            config['fetch_params']['tickers'],
            config['fetch_params']['start_date'],
            config['fetch_params']['end_date']
        )
        logger.info("SEC data fetched")

        reuters_df = crawl_reuters_news(
            config['fetch_params']['tickers'],
            config['fetch_params']['start_date'],
            config['fetch_params']['end_date']
        )
        logger.info("Reuters news crawled")

        news_file_path = f"{config['data_paths']['news_data']}news.parquet"
        news_df = load_news_data(news_file_path)
        logger.info("Existing news data loaded")

        combined_news_df = combine_news_data(news_df, reuters_df, news_file_path)
        logger.info("News data combined and saved")

        price_data_path = f"{config['data_paths']['price_data']}price.pkl"
        price_dict = pd.read_pickle(price_data_path)
        price_df = pd.DataFrame.from_dict(price_dict, orient='index')
        price_df.index.name = 'date'
        price_df = price_df.reset_index()
        trading_dates = price_df['date'].tolist()

        processed_news_path = f"{config['data_paths']['news_data']}concatenated_news_filtered.parquet"
        news_data = pd.read_parquet(news_file_path)
        process_reuters_dates(news_data, trading_dates, processed_news_path)
        logger.info("Reuters dates processed and saved")

        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()
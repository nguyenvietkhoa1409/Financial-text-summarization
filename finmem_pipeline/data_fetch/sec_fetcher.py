# dependencies
import os
import httpx
import pytz
import time
import itertools
import polars as pl
import yaml
from cleantext import clean
from tqdm import tqdm
from rich import print
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from httpx import RequestError
from datetime import datetime
from dateutil import parser
from typing import List
from finmem_pipeline.utils.logging import setup_logging

# set up logger
logger = setup_logging()

# load environment variables
logger.info("Program starts")
print(load_dotenv(os.path.join(".env")))

# load configuration from YAML
with open(os.path.join("config", "config.yaml"), "r") as file:
    config = yaml.safe_load(file)

# parameters from config.yaml
NUM_WORKERS = 1
ENDPOINT_URL = "https://api.sec-api.io?token={SEC_KEY}"
EXTRACTOR_URL = "https://api.sec-api.io/extractor?url={url}&token={SEC_KEY}&item={item}"
TEN_K_ITEM_CODE = [
    "7",  # target
]
TEN_Q_ITEM_CODE = [
    "part1item2",  # target
]
EIGHT_K_ITEM_CODE = [
    "1-1", "1-2", "1-3", "1-4",
    "2-1", "2-2", "2-3", "2-4", "2-5", "2-6",
    "3-1", "3-2", "3-3",
    "4-1", "4-2",
    "5-1", "5-2", "5-3", "5-4", "5-5", "5-6", "5-7", "5-8",
    "6-1", "6-2", "6-3", "6-4", "6-5", "6-6", "6-10",
    "7-1", "8-1",
]
SIZE = 50
SLEEP_TIME = 10
START_DATE = config["fetch_params"]["start_date"]  # '2023-08-17'
END_DATE = config["fetch_params"]["end_date"]      # '2025-04-10'
unique_equities = config["fetch_params"]["tickers"] # ['TSLA', 'AMZN', 'MSFT', 'NFLX']
OUTPUT_PATH = os.path.join(config["data_paths"]["news_data"], "filing_data.parquet")

# convert time zone
def convert_utc_to_est(utc_dt: datetime) -> datetime:
    utc = pytz.UTC
    utc_dt = utc.localize(utc_dt)
    est = pytz.timezone("US/Eastern")
    return utc_dt.astimezone(est).replace(tzinfo=None)

# get index table
def get_index_single(symbol: str, type: str) -> pl.DataFrame:
    logger.info(f"Fetching index data for {symbol} (form type: {type})")
    with httpx.Client() as client:
        page_count = 0
        ticker_list = []
        cik_list = []
        timestamp_list = []
        document_url_list = []

        while True:
            query_payload = {
                "query": {
                    "query_string": {"query": f'ticker:{symbol} AND formType:"{type}"'}
                },
                "from": f"{page_count}",
                "size": f"{SIZE}",
                "sort": [{"filedAt": {"order": "desc"}}],
            }
            # whether to break
            try:
                response = client.post(
                    ENDPOINT_URL.format(SEC_KEY=os.environ.get("SEC_KEY")),
                    json=query_payload,
                )
                if response.status_code != 200:
                    if response.status_code != 429:
                        logger.error(f"Request failed for {symbol} (type: {type}): {response.text}")
                        raise RequestError(response.text)
                    logger.info("Hit rate limit, retrying after delay")
                    time.sleep(SLEEP_TIME)
                    response = client.post(
                        ENDPOINT_URL.format(SEC_KEY=os.environ.get("SEC_KEY")),
                        json=query_payload,
                    )
                if response.status_code != 200:
                    logger.error(f"Request failed after retry for {symbol} (type: {type}): {response.text}")
                    raise RequestError(response.text)
                result = response.json()
                if len(result["filings"]) == 0:
                    break
                # parse data
                for cur_record in result["filings"]:
                    for cur_document in cur_record["documentFormatFiles"]:
                        if cur_document["type"] == type:
                            document_url_list.append(cur_document["documentUrl"])
                            ticker_list.append(cur_record["ticker"])
                            cik_list.append(cur_record["cik"])
                            timestamp_list.append(
                                parser.parse(cur_record["filedAt"]).replace(tzinfo=None)
                            )
                            break

                # page count
                page_count += 1 * SIZE

            except Exception as e:
                logger.error(f"Failed to fetch index data for {symbol} (type: {type}): {e}")
                raise

        df = pl.DataFrame(
            {
                "ticker": ticker_list,
                "cik": cik_list,
                "utc_timestamp": timestamp_list,
                "document_url": document_url_list,
            }
        )
        utc_times = df["utc_timestamp"].to_list()
        est_times = [convert_utc_to_est(t) for t in utc_times]
        df = df.with_columns(
            pl.Series(est_times).alias("est_timestamp"), pl.lit(type).alias("type")
        )

        return df

def get_index(symbols: List[str], type: str) -> pl.DataFrame:
    index_params = [{"symbol": e, "type": type} for e in symbols]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        index_data = list(
            tqdm(
                executor.map(lambda x: get_index_single(**x), index_params),
                total=len(index_params),
                desc=f"Downloading index data for {type}",
            )
        )
    index_df = pl.concat([f for f in index_data if f.shape[0] > 0])
    index_df = index_df.unique(subset="document_url")
    return index_df

def request_content_single(section: str, file_url: str) -> str:
    try:
        ret_txt = ""
        cur_extractor_url = EXTRACTOR_URL.format(
            url=file_url, item=section, SEC_KEY=os.environ.get("SEC_KEY")
        )
        with httpx.Client() as client:
            response = client.get(cur_extractor_url)
            if response.status_code != 200:
                if response.status_code != 429:
                    logger.error(f"Request failed for section {section}, URL {file_url}: {response.text}")
                    raise ValueError(response.text)
                logger.info("Hit rate limit, retrying after delay")
                response = client.get(cur_extractor_url)
            if response.status_code != 200:
                logger.error(f"Request failed after retry for section {section}, URL {file_url}: {response.text}")
                raise ValueError(response.text)
        ret_txt += response.text
        return clean(
            ret_txt,
            fix_unicode=True,
            to_ascii=True,
            lower=True,
            no_line_breaks=True,
        )
    except Exception as e:
        logger.error(f"Failed to fetch content for section {section}, URL {file_url}: {e}")
        return None

def request_content(filings: List[str], sections: List[str]) -> List[str]:
    content_params = [
        {"section": cur_section, "file_url": e}
        for e, cur_section in itertools.product(filings, sections)
    ]
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        content_data = list(
            tqdm(
                executor.map(lambda x: request_content_single(**x), content_params),
                total=len(content_params),
                desc="Downloading content data",
            )
        )
    return content_data

if __name__ == "__main__":
    # get file index
    ten_k_index_table = get_index(unique_equities, "10-K")
    ten_q_index_table = get_index(unique_equities, "10-Q")
    # eight_k_index_table = get_index(unique_equities, "8-K")

    # request content
    ten_k_df = (
        pl.DataFrame(
            [
                pl.Series("document_url", ten_k_index_table["document_url"].to_list()),
                pl.Series(
                    "content",
                    request_content(
                        filings=ten_k_index_table["document_url"].to_list(),
                        sections=TEN_K_ITEM_CODE,
                    ),
                ),
            ]
        )
        .join(ten_k_index_table, on="document_url")
        .drop_nulls()
    )
    ten_q_df = (
        pl.DataFrame(
            [
                pl.Series("document_url", ten_q_index_table["document_url"].to_list()),
                pl.Series(
                    "content",
                    request_content(
                        filings=ten_q_index_table["document_url"].to_list(),
                        sections=TEN_Q_ITEM_CODE,
                    ),
                ),
            ]
        )
        .join(ten_q_index_table, on="document_url")
        .drop_nulls()
    )
    # eight_k_df = (
    #     pl.DataFrame(
    #         [
    #             pl.Series(
    #                 "document_url", eight_k_index_table["document_url"].to_list()
    #             ),
    #             pl.Series(
    #                 "content",
    #                 request_content(
    #                     filings=eight_k_index_table["document_url"].to_list(),
    #                     sections=EIGHT_K_ITEM_CODE,
    #                 ),
    #             ),
    #         ]
    #     )
    #     .join(eight_k_index_table, on="document_url")
    #     .drop_nulls()
    # )

    # filing_data = pl.concat([ten_k_df, ten_q_df, eight_k_df])
    filing_data = pl.concat([ten_k_df, ten_q_df])
    filing_data.write_parquet(OUTPUT_PATH)
    logger.info("Program ends")
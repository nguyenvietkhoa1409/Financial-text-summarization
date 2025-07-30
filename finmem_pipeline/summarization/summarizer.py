import os
import requests
import pandas as pd
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from finmem_pipeline.utils.logging import setup_logging
from finmem_pipeline.config import load_config

logger = setup_logging()
config = load_config()

def format_prompt_for_investor(article_text, symbol):
    """
    Formats the prompt for investor-tailored summarization.

    Args:
        article_text (str): The news article text.
        symbol (str): The stock ticker symbol.

    Returns:
        str: Formatted prompt for the LLM.
    """
    return f"""
For the stock {symbol}, summarize the following news article in no more than 140 words. Focus on key factual events, quantitative figures (e.g., earnings, percentage changes, dates), material risks, relevant corporate actions, regulatory developments, or market sentiment that could impact {symbol}'s stock price. Ensure all information is directly from the article and accurately reported, especially numerical figures. Avoid speculation, assumptions, or vague generalizations. Do not include opinions or inferred conclusions not directly supported by the source text.

Article:
{article_text}

Summary:
"""

def summarize_text(idx, text, symbol, prompt_formatter, headers, endpoint_url, max_retries=5, wait_time=10):
    """
    Summarizes a single article using the LLM endpoint with retry logic.

    Args:
        idx (int): Index of the article in the DataFrame.
        text (str): Article content.
        symbol (str): Stock ticker symbol.
        prompt_formatter (callable): Function to format the prompt.
        headers (dict): HTTP headers for the API request.
        endpoint_url (str): LLM endpoint URL.
        max_retries (int): Maximum number of retries for API calls.
        wait_time (int): Wait time between retries in seconds.

    Returns:
        tuple: (index, summary) or (index, None) if failed.
    """
    full_prompt = prompt_formatter(text, symbol)
    data = {
        "inputs": full_prompt,
        "parameters": {
            "max_new_tokens": 220,
            "do_sample": False,
            "temperature": 0.5,
        }
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(endpoint_url, headers=headers, json=data, timeout=60)
            if response.status_code == 200:
                result = response.json()
                summary = result[0]["generated_text"] if isinstance(result, list) else result["generated_text"]
                logger.info(f"Generated summary for index {idx}")
                return idx, summary
            elif response.status_code in [429, 502, 503]:
                logger.warning(f"Retry {attempt+1} for index {idx} - status {response.status_code}")
            else:
                logger.error(f"Error {response.status_code} for index {idx}: {response.text}")
        except Exception as e:
            logger.error(f"Exception at index {idx}: {e}")
        time.sleep(wait_time)
    logger.error(f"Failed to summarize index {idx} after {max_retries} retries")
    return idx, None

def summarize_batch(batch_df, prompt_formatter, headers, endpoint_url, batch_size):
    """
    Summarizes a batch of articles using threading.

    Args:
        batch_df (pd.DataFrame): Batch of articles to summarize.
        prompt_formatter (callable): Function to format the prompt.
        headers (dict): HTTP headers for the API request.
        endpoint_url (str): LLM endpoint URL.
        batch_size (int): Number of concurrent threads.

    Returns:
        list: List of (index, summary) tuples.
    """
    with ThreadPoolExecutor(max_workers=batch_size) as executor:
        futures = {
            executor.submit(summarize_text, row.Index, row.content, row.equity, prompt_formatter, headers, endpoint_url): row.Index
            for row in batch_df.itertuples()
        }
        results = []
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as exc:
                logger.error(f"Index {futures[future]} generated an exception: {exc}")
                results.append((futures[future], None))
        return results

def generate_summaries(news_df, save_path, batch_size, headers, endpoint_url):
    """
    Generates summaries for all articles and saves checkpoints.

    Args:
        news_df (pd.DataFrame): DataFrame with news articles.
        save_path (str): Path to save checkpoint parquet file.
        batch_size (int): Number of articles to process concurrently.
        headers (dict): HTTP headers for the API request.
        endpoint_url (str): LLM endpoint URL.

    Returns:
        pd.DataFrame: DataFrame with summaries in 'hf_summary' column.
    """
    try:
        news_df = news_df.copy()
        if 'hf_summary' not in news_df.columns:
            news_df['hf_summary'] = None

        start_index = 0
        if os.path.exists(save_path):
            checkpoint_df = pd.read_parquet(save_path)
            last_processed_index = checkpoint_df['hf_summary'].last_valid_index()
            if last_processed_index is not None:
                start_index = last_processed_index + 1
                logger.info(f"Resuming summarization from index {start_index}")

        total_records = len(news_df)
        for i in tqdm(range(start_index, total_records, batch_size), desc="Summarizing"):
            batch_end = min(i + batch_size, total_records)
            batch_df = news_df.iloc[i:batch_end]
            batch_results = summarize_batch(batch_df, format_prompt_for_investor, headers, endpoint_url, batch_size)

            for idx, summary in batch_results:
                news_df.loc[idx, "hf_summary"] = summary

            news_df.to_parquet(save_path, index=False)
            logger.info(f"Checkpoint saved to {save_path} at index {batch_end}")

        logger.info(f"Summarization completed, saved to {save_path}")
        return news_df
    except Exception as e:
        logger.error(f"Failed to generate summaries: {e}")
        raise
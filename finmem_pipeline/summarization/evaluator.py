import pandas as pd
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
from finmem_pipeline.utils.logging import setup_logging

logger = setup_logging()

def extract_summary(text):
    """
    Extracts the summary portion from the LLM-generated text.

    Args:
        text: The string from the 'hf_summary' column.

    Returns:
        The summary part of the string, or the original text if 'Summary:' is not found
        or the input is not a string.
    """
    if not isinstance(text, str):
        return text
    summary_start = text.find("Summary:")
    return text[summary_start + len("Summary:"):].strip() if summary_start != -1 else text

def evaluate_summaries(news_df, model_name='all-MiniLM-L6-v2', batch_size=32, threshold=0.4):
    """
    Evaluates summaries using SBERT and filters those below the threshold.

    Args:
        news_df (pd.DataFrame): DataFrame with 'content', 'hf_summary', and 'summary' columns.
        model_name (str): SBERT model name.
        batch_size (int): Batch size for encoding.
        threshold (float): Minimum similarity score to keep a summary.

    Returns:
        pd.DataFrame: Filtered DataFrame with updated 'summary' and 'similarity_score' columns.
    """
    try:
        news_df = news_df.copy()
        # Overwrite the 'summary' column with extracted summaries
        news_df['summary'] = news_df['hf_summary'].apply(extract_summary)

        model = SentenceTransformer(model_name)
        source_texts = news_df['content'].tolist()
        summaries = news_df['summary'].tolist()

        source_embeddings = model.encode(source_texts, batch_size=batch_size, show_progress_bar=True)
        summary_embeddings = model.encode(summaries, batch_size=batch_size, show_progress_bar=True)

        similarities = [model.similarity(src, summ).item() for src, summ in zip(source_embeddings, summary_embeddings)]
        news_df['similarity_score'] = similarities

        filtered_df = news_df[news_df['similarity_score'] >= threshold]
        logger.info(f"Filtered {len(filtered_df)} out of {len(news_df)} summaries with threshold {threshold}")

        return filtered_df
    except Exception as e:
        logger.error(f"Failed to evaluate summaries: {e}")
        raise

def analyze_token_lengths(news_df, model_name='meta-llama/Llama-3.2-3B-Instruct', output_dir=None):
    """
    Analyzes token lengths of content and summaries, generating histograms and boxplots.

    Args:
        news_df (pd.DataFrame): DataFrame with 'content' and 'summary' columns.
        model_name (str): Tokenizer model name.
        output_dir (str, optional): Directory to save plots. If None, plots are not saved.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        c_len = [len(tokenizer.encode(c)) for c in news_df['content'] if isinstance(c, str)]
        s_len = [len(tokenizer.encode(s)) for s in news_df['summary'] if isinstance(s, str)]

        # Create plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Content length histogram
        axes[0].hist(c_len, bins=50, color="#1f77b4", edgecolor="black", alpha=0.8)
        axes[0].set_title("News Token Length Distribution", fontsize=14)
        axes[0].set_xlabel("Token Length", fontsize=12)
        axes[0].set_ylabel("Count", fontsize=12)
        axes[0].grid(True, linestyle='--', alpha=0.6)

        # Summary length histogram
        axes[1].hist(s_len, bins=50, color="#ff7f0e", edgecolor="black", alpha=0.8)
        axes[1].set_title("Summary Token Length Distribution", fontsize=14)
        axes[1].set_xlabel("Token Length", fontsize=12)
        axes[1].grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "token_length_histograms.png"))
        plt.close()

        # Boxplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].boxplot(c_len, patch_artist=True, boxprops=dict(facecolor="#1f77b4", color='black'),
                        medianprops=dict(color='red'), whiskerprops=dict(color='black'),
                        capprops=dict(color='black'), flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, linestyle='none'))
        axes[0].set_title("Content Token Length", fontsize=14)
        axes[0].set_ylabel("Token Length", fontsize=12)
        axes[0].grid(axis='y', linestyle='--', alpha=0.6)

        axes[1].boxplot(s_len, patch_artist=True, boxprops=dict(facecolor="#ff7f0e", color='black'),
                        medianprops=dict(color='red'), whiskerprops=dict(color='black'),
                        capprops=dict(color='black'), flierprops=dict(marker='o', markerfacecolor='orange', markersize=5, linestyle='none'))
        axes[1].set_title("Summary Token Length", fontsize=14)
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)

        plt.tight_layout()
        if output_dir:
            plt.savefig(os.path.join(output_dir, "token_length_boxplots.png"))
        plt.close()

        logger.info("Token length analysis completed")
    except Exception as e:
        logger.error(f"Failed to analyze token lengths: {e}")
        raise
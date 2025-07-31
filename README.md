# Financial text summarization project 📈

Welcome to **Financial-text-summarization**! 🚀 This project is a powerful, modular data pipeline designed to aggregate, process, and analyze financial data, delivering investor-focused insights with efficiency and precision. Built upon the innovative **FinMem** framework, it combines multiple data sources into a unified schema, leverages cost-effective LLM inference, and ensures high-quality results through robust evaluation and visualization techniques.

## 🎯 Purpose

This project aims to empower financial analysts, data scientists, and developers by:
- 🌐 **Aggregating** diverse financial data (stock prices, news articles, and SEC filings) into a cohesive dataset.
- ✍️ **Summarizing** large-scale textual data (~24,000 records, ~2,000 tokens each) into concise, investor-tailored insights.
- ✅ **Evaluating** summary quality using semantic analysis to ensure accuracy and relevance.
- 📊 **Visualizing** data metrics to validate consistency and performance.

This pipeline is optimized for scalability and cost-efficiency, using the **" $6.75 llama-3-2-3b-instruct-xal"** [HF inference-endpoints](https://huggingface.co/inference-endpoints/dedicated) (Nvidia A10G) as a low-cost alternative to expensive APIs like GPT, making it ideal for processing large datasets.

## 🛠️ Core Modules, Functions, and Results

The pipeline is built with a modular architecture, ensuring flexibility and robustness. Here’s how it works:

### Core Modules & Steps
1. **📥 Data Fetching** (`finmem_pipeline/data_fetch/`):
   - **Yahoo Finance** (`yfinance_fetcher.py`): Retrieves historical stock price data.
   - **Alpaca API** (`alpaca_fetcher.py`): Fetches stock and news data.
   - **SEC Filings** (`sec_fetcher.py`): Placeholder for retrieving SEC filings (to be implemented).
   - **Reuters News** (`reuters_crawler.py`): Crawls news articles from Reuters.

2. **🔄 Data Processing** (`finmem_pipeline/data_processing/`):
   - **Price Processing** (`price_processor.py`): Cleans and merges stock price data.
   - **News Processing** (`news_processor.py`): Combines news from multiple sources.
   - **Date Adjustment** (`reuters_date_adjuster.py`): Aligns news timestamps with trading days.

3. **📝 Summarization** (`finmem_pipeline/summarization/summarizer.py`):
   - Generates investor-focused summaries using the cost-effective **Llama-3-2-3B-Instruct** endpoint.
   - Processes ~24,000 records (~2,000 tokens each) with threading, checkpointing, and retry logic for reliability.

4. **✅ Evaluation & Visualization** (`finmem_pipeline/summarization/evaluator.py`):
   - Evaluates summaries using **SBERT** (Sentence-BERT) for semantic similarity, filtering out low-quality summaries (score < 0.4).
   - Visualizes token length distributions for raw content and summaries using histograms and boxplots.

### Workflow
1. **Fetch**: Collect raw data from Yahoo Finance, Alpaca, Reuters, and (future) SEC sources.
2. **Process**: Merge and clean data into a unified schema.
3. **Summarize**: Generate concise summaries with the Llama-3-2-3B-Instruct endpoint.
4. **Evaluate**: Filter summaries based on SBERT similarity scores.
5. **Visualize**: Analyze token lengths to ensure quality and consistency.
6. **Output**: Save a refined dataset with high-quality summaries and visualizations.

### Results
- 📊 A unified dataset combining stock prices, news, and (future) SEC filings.
- ✍️ ~24,000 high-quality summaries tailored for investors.
- 📈 Visualizations of token length distributions and quality metrics.

## 📂 Project Structure

The project is organized for clarity and maintainability:
```bash
finmem_pipeline/
├── finmem_pipeline/           # Core Python package
│   ├── __init__.py            # Package initialization
│   ├── config.py              # Loads configuration settings
│   ├── data_fetch/            # Data fetching modules
│   │   ├── __init__.py
│   │   ├── yfinance_fetcher.py
│   │   ├── alpaca_fetcher.py
│   │   ├── sec_fetcher.py
│   │   └── reuters_crawler.py
│   ├── data_processing/       # Data processing modules
│   │   ├── __init__.py
│   │   ├── price_processor.py
│   │   ├── news_processor.py
│   │   └── reuters_date_adjuster.py
│   ├── utils/                 # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py         # Logging setup
│   │   └── storage.py         # Data storage utilities
│   ├── summarization/         # Summarization and evaluation
│   │   ├── __init__.py
│   │   ├── summarizer.py      # LLM-based summarization
│   │   └── evaluator.py       # SBERT evaluation and visualization
├── scripts/                   # Executable scripts
│   └── run_pipeline.py        # Main pipeline script
├── config/                    # Configuration files
│   └── config.yaml            # Pipeline configuration
├── .env                       # Environment variables (API keys)
├── README.md                  # Project documentation
└── requirements.txt           # Dependency list

```
## 🌟 Key Highlights

- **Unified Data Pipeline** 🌐: Seamlessly aggregates data from Yahoo Finance, Alpaca, Reuters, and (future) SEC filings into a single, structured schema for comprehensive financial analysis.
- **Robust Logic** 🔧: Incorporates error handling, retry mechanisms, and detailed logging for reliable operation, even with large datasets.
- **Cost-Effective LLM Inference** 💸: Uses the **"Endpoints $6.75 llama-3-2-3b-instruct-xal"** (Nvidia A10G) to process ~24,000 records (~2,000 tokens each) at a fraction of the cost of GPT APIs.
- **Quality Assurance** ✅: Employs SBERT for semantic evaluation and filters low-quality summaries (score < 0.4). Visualizes token length distributions to ensure consistency and reliability.

## 🎓 Attribution to FinMem Framework

This project builds upon the **FinMem** framework, as described in the paper *FINMEM: A Performance-Enhanced LLM Trading Agent with Layered Memory and Character Design* ([arXiv:2311.13743](https://arxiv.org/abs/2311.13743)) by Yu et al. The FinMem framework provides a robust foundation for LLM-based financial agents with layered memory and character design.
**My Contribution**:
- **Cost-Effective LLM Inference**: Integrated the **Llama-3-2-3B-Instruct** endpoint, significantly reducing costs compared to GPT APIs while maintaining high-quality summarization for large-scale datasets (~24,000 records, ~2,000 tokens each).
- **Evaluation & Visualization**: Added SBERT-based semantic evaluation to filter low-quality summaries and implemented visualization of token length distributions to ensure result quality and consistency.

## 🚀 Getting Started

### Prerequisites
- 🐍 Python 3.8+
- 🔑 API keys for Alpaca and Hugging Face
- 📦 Dependencies listed in `requirements.txt`

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/finmem-pipeline.git
   cd finmem-pipeline


Install Dependencies:pip install -r requirements.txt


Configure Environment:Create a .env file in the project root:ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
GOOGLE_DRIVE_CREDENTIALS_PATH=/path/to/your/credentials.json
HF_TOKEN=your_huggingface_token


Run the Pipeline:python scripts/run_pipeline.py



🛠️ Usage
Execute scripts/run_pipeline.py to:

Fetch and process financial data from multiple sources.
Generate investor-tailored summaries using the Llama-3-2-3B-Instruct endpoint.
Evaluate summaries with SBERT and visualize token metrics.
Save results to parquet files and generate visualizations.

🤝 Contributing
We welcome contributions! 🌟 Please fork the repository, create a pull request, or open an issue to suggest improvements or report bugs.

🙏 Acknowledgments
FinMem Team: For their groundbreaking work on the FinMem framework, providing a robust foundation for this pipeline.
Hugging Face: For providing the cost-effective Llama-3-2-3B-Instruct endpoint.
Open Source Community: For the libraries and tools that power this project.

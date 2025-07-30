import os
import yaml
from dotenv import load_dotenv
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    """
    Load configuration from YAML file and environment variables.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration dictionary.
    """
    load_dotenv()
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    config['alpaca_api_key'] = os.getenv('ALPACA_API_KEY')
    config['alpaca_secret_key'] = os.getenv('ALPACA_SECRET_KEY')
    config['google_drive_credentials'] = os.getenv('GOOGLE_DRIVE_CREDENTIALS_PATH')
    
    return config

def ensure_directories(config: dict):
    """
    Ensure that required directories exist.
    
    Args:
        config (dict): Configuration dictionary.
    """
    for path in [config['data_paths']['price_data'], config['data_paths']['news_data']]:
        Path(path).mkdir(parents=True, exist_ok=True)
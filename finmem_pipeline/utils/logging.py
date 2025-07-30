import logging
import os
from pathlib import Path

def setup_logging(log_dir: str = "logs"):
    """
    Configure logging for the pipeline.

    Args:
        log_dir (str): Directory to store log files.
    """
    Path(log_dir).mkdir(exist_ok=True)
    log_file = os.path.join(log_dir, "pipeline.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)
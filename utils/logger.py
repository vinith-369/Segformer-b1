# ---------------------------------------------------------------
# Logging utility: file + console output
# ---------------------------------------------------------------

import os
import logging
from datetime import datetime


def setup_logger(log_dir='logs', log_name=None):
    """Set up logging to both file and console.

    Args:
        log_dir: Directory to save log files.
        log_name: Custom log filename. If None, auto-generates with timestamp.

    Returns:
        logger: Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)

    if log_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_name = f'segformer_b1_{timestamp}.log'

    log_path = os.path.join(log_dir, log_name)

    # Clear any existing handlers
    logger = logging.getLogger('segformer')
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # File handler
    fh = logging.FileHandler(log_path, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    logger.info(f"Logging to: {log_path}")
    return logger


def get_logger():
    """Get the segformer logger."""
    return logging.getLogger('segformer')

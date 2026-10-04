"""
Logging Setup Utility
Provides structured logging to console and optional log files.
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def get_logger(name: str = "dna_pipeline", log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Get or create a configured logger.

    Args:
        name: Name of the logger.
        log_file: Optional file path to write log output.
        level: Logging level (e.g. logging.INFO, logging.DEBUG).

    Returns:
        logging.Logger: Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already created
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger

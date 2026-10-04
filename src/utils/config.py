"""
Configuration Management Utility
Loads YAML configuration with validation, directory auto-creation, and parameter access.
"""
from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # Assuming config.py is at <root>/src/utils/config.py
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load project configuration from a YAML file.

    Args:
        config_path: Relative or absolute path to config.yaml.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / config_path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Ensure critical directory paths exist
    ensure_directories(config)

    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    """Ensure that standard output and data directories defined in config exist."""
    root = get_project_root()

    dir_keys = [
        ("data", "raw_dir"),
        ("data", "processed_dir"),
        ("data", "external_dir"),
        ("paths", "results_dir"),
        ("paths", "metrics_dir"),
        ("paths", "predictions_dir"),
        ("paths", "figures_dir"),
        ("paths", "reports_dir"),
        ("paths", "models_dir"),
    ]

    for section, key in dir_keys:
        if section in config and key in config[section]:
            dir_path = root / config[section][key]
            dir_path.mkdir(parents=True, exist_ok=True)

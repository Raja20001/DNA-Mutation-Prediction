"""
Dataset Loader and Serializer Module
Handles loading of DNA variant CSV datasets, schema inspection,
and saving processed data and artifacts.
"""
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd

from ..utils.logger import get_logger
from .validator import validate_dataframe

logger = get_logger("data_loader")


def detect_delimiter(file_path: Union[str, Path], num_lines: int = 5) -> str:
    """
    Detect whether a file is separated by comma, tab, semicolon, or whitespace.

    Args:
        file_path: Path to the delimited file.
        num_lines: Number of lines to inspect.

    Returns:
        str: Detected delimiter (default: ',').
    """
    candidates = [",", "\t", ";", "|"]
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        sample_lines = [f.readline() for _ in range(num_lines)]

    counts = {c: 0 for c in candidates}
    for line in sample_lines:
        for c in candidates:
            counts[c] += line.count(c)

    best_cand = max(counts, key=counts.get)
    return best_cand if counts[best_cand] > 0 else ","


def load_dataset(
    file_path: Union[str, Path],
    delimiter: Optional[str] = None,
    required_columns: Optional[list] = None,
    allowed_bases: Optional[set] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a DNA variant dataset from a CSV/TSV file and produce an initial validation summary.

    Args:
        file_path: Path to dataset.
        delimiter: Optional explicit delimiter.
        required_columns: Columns required to be present (default: ['sequence', 'label']).
        allowed_bases: Nucleotide base set (default: A, C, G, T).

    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]:
            - df: Loaded pandas DataFrame.
            - summary: Inspection and validation metrics report.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path.resolve()}")

    sep = delimiter or detect_delimiter(path)
    logger.info(f"Loading dataset from '{path.name}' using delimiter: '{sep}'")

    try:
        df = pd.read_csv(path, sep=sep, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, sep=sep, encoding="latin-1")

    # Standardize column names (strip whitespace and convert to lowercase)
    df.columns = [str(c).strip().lower() for c in df.columns]

    # Validate dataframe and inspect schema
    summary = validate_dataframe(
        df=df,
        required_columns=required_columns,
        allowed_bases=allowed_bases if allowed_bases else {"A", "C", "G", "T"},
    )

    logger.info(
        f"Loaded {summary['total_records']} rows, {summary['total_columns']} cols. "
        f"Available supported fields: {summary['available_supported_columns']}"
    )

    return df, summary


def save_processed_dataset(
    df: pd.DataFrame,
    output_path: Union[str, Path],
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save a cleaned and preprocessed dataset along with accompanying JSON metadata.

    Args:
        df: DataFrame to save.
        output_path: Filepath where CSV should be written.
        metadata: Optional metadata dictionary to save as .json alongside the CSV.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out, index=False)
    logger.info(f"Saved processed dataset ({len(df)} rows) to: {out}")

    if metadata:
        meta_path = out.with_suffix(".json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved processing metadata to: {meta_path}")

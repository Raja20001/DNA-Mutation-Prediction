"""
DNA Sequence and Dataset Validation Module
Validates DNA sequences, detects IUPAC ambiguity, inspects dataset schemas,
and checks for class imbalance and data quality issues.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

# Standard DNA nucleotides
STANDARD_NUCLEOTIDES: Set[str] = {"A", "C", "G", "T"}

# IUPAC ambiguity nucleotide codes (excluding standard A, C, G, T)
IUPAC_AMBIGUITY_CODES: Dict[str, str] = {
    "R": "A or G (puRine)",
    "Y": "C or T (pYrimidine)",
    "S": "G or C (Strong)",
    "W": "A or T (Weak)",
    "K": "G or T (Keto)",
    "M": "A or C (aMino)",
    "B": "C, G or T (not A)",
    "D": "A, G or T (not C)",
    "H": "A, C or T (not G)",
    "V": "A, C or G (not T)",
    "N": "A, C, G or T (aNy)",
}

# Standard expected optional genomic annotation columns
SUPPORTED_COLUMNS: List[str] = [
    "sequence",
    "label",
    "variant_id",
    "chromosome",
    "position",
    "reference",
    "alternate",
    "gene",
    "transcript",
    "mutation_type",
    "condition",
    "hgvs",
]


def validate_sequence(
    sequence: Any,
    allowed_bases: Set[str] = STANDARD_NUCLEOTIDES,
    allow_ambiguous: bool = False,
    min_length: int = 1,
    max_length: int = 50000,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    """
    Validate a single DNA sequence string.

    Args:
        sequence: The DNA sequence (string or convertible to string).
        allowed_bases: Set of permissible base characters (default: {'A', 'C', 'G', 'T'}).
        allow_ambiguous: If True, IUPAC ambiguity codes are considered valid.
        min_length: Minimum allowable sequence length.
        max_length: Maximum allowable sequence length.

    Returns:
        Tuple[bool, List[str], Dict[str, Any]]:
            - is_valid: True if sequence passes all checks.
            - error_messages: List of reasons if invalid.
            - details: Dictionary with length, invalid characters and their 0-based indices,
                       and detected IUPAC ambiguous characters.
    """
    errors: List[str] = []
    details: Dict[str, Any] = {
        "length": 0,
        "invalid_characters": {},
        "ambiguous_characters": {},
        "is_empty": False,
    }

    if sequence is None or (isinstance(sequence, float) and np.isnan(sequence)):
        errors.append("Sequence is null/NaN.")
        details["is_empty"] = True
        return False, errors, details

    seq_str = str(sequence).strip().upper()
    seq_len = len(seq_str)
    details["length"] = seq_len

    if seq_len == 0:
        errors.append("Sequence is empty.")
        details["is_empty"] = True
        return False, errors, details

    if seq_len < min_length:
        errors.append(f"Sequence length ({seq_len}) is below minimum ({min_length}).")

    if seq_len > max_length:
        errors.append(f"Sequence length ({seq_len}) exceeds maximum ({max_length}).")

    # Detect invalid characters and IUPAC codes
    valid_set = set(allowed_bases)
    if allow_ambiguous:
        valid_set = valid_set.union(set(IUPAC_AMBIGUITY_CODES.keys()))

    invalid_chars: Dict[str, List[int]] = {}
    ambiguous_chars: Dict[str, List[int]] = {}

    for idx, char in enumerate(seq_str):
        if char not in valid_set:
            if char not in invalid_chars:
                invalid_chars[char] = []
            invalid_chars[char].append(idx)

        if char in IUPAC_AMBIGUITY_CODES:
            if char not in ambiguous_chars:
                ambiguous_chars[char] = []
            ambiguous_chars[char].append(idx)

    details["invalid_characters"] = invalid_chars
    details["ambiguous_characters"] = ambiguous_chars

    if invalid_chars:
        chars_summary = ", ".join(f"'{c}' ({len(idxs)} times)" for c, idxs in invalid_chars.items())
        errors.append(f"Invalid non-DNA characters found: {chars_summary}.")

    if not allow_ambiguous and ambiguous_chars:
        ambig_summary = ", ".join(f"'{c}' ({len(idxs)} times)" for c, idxs in ambiguous_chars.items())
        errors.append(f"Ambiguous IUPAC bases found when allow_ambiguous=False: {ambig_summary}.")

    is_valid = len(errors) == 0
    return is_valid, errors, details


def compute_sequence_stats(sequences: pd.Series) -> Dict[str, Any]:
    """
    Compute descriptive statistics for a series of DNA sequences.

    Args:
        sequences: Pandas Series of DNA sequence strings.

    Returns:
        Dict[str, Any]: Summary statistics including min, max, mean, median, standard deviation.
    """
    clean_series = sequences.dropna().astype(str).str.strip()
    lengths = clean_series.str.len()

    if len(lengths) == 0:
        return {
            "total_count": 0,
            "min_length": 0,
            "max_length": 0,
            "mean_length": 0.0,
            "median_length": 0.0,
            "std_length": 0.0,
        }

    return {
        "total_count": int(len(clean_series)),
        "min_length": int(lengths.min()),
        "max_length": int(lengths.max()),
        "mean_length": float(round(lengths.mean(), 2)),
        "median_length": float(round(lengths.median(), 2)),
        "std_length": float(round(lengths.std(ddof=0), 2)),
    }


def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    allowed_bases: Set[str] = STANDARD_NUCLEOTIDES,
    allow_ambiguous: bool = False,
    min_length: int = 10,
    max_length: int = 5000,
) -> Dict[str, Any]:
    """
    Inspect an uploaded DataFrame and dynamically determine available columns,
    compute sequence quality metrics, check duplicates, missing values, and class distribution.

    Args:
        df: Input DataFrame.
        required_columns: Columns that must be present (default: ['sequence', 'label']).
        allowed_bases: Permissible bases (default: A, C, G, T).
        allow_ambiguous: Whether IUPAC ambiguous bases are permissible.
        min_length: Minimum valid sequence length.
        max_length: Maximum valid sequence length.

    Returns:
        Dict[str, Any]: Comprehensive inspection and validation report.
    """
    if required_columns is None:
        required_columns = ["sequence", "label"]

    total_records = len(df)
    total_columns = len(df.columns)
    columns_present = list(df.columns)

    # Dynamic column discovery
    missing_required = [col for col in required_columns if col not in df.columns]
    available_supported = [col for col in SUPPORTED_COLUMNS if col in df.columns]
    unsupported_columns = [col for col in df.columns if col not in SUPPORTED_COLUMNS]

    # Missing values inspection
    missing_values = df.isnull().sum().to_dict()

    # Duplicate records
    exact_duplicates = int(df.duplicated().sum())
    sequence_duplicates = int(df.duplicated(subset=["sequence"]).sum()) if "sequence" in df.columns else 0

    # Class distribution and imbalance check
    class_distribution: Dict[str, int] = {}
    class_percentages: Dict[str, float] = {}
    class_imbalance_ratio = 1.0
    is_imbalanced = False

    if "label" in df.columns and total_records > 0:
        counts = df["label"].value_counts(dropna=False).to_dict()
        class_distribution = {str(k): int(v) for k, v in counts.items()}
        class_percentages = {
            str(k): float(round((v / total_records) * 100, 2)) for k, v in counts.items()
        }
        if len(counts) > 1:
            max_c = max(counts.values())
            min_c = min(counts.values())
            class_imbalance_ratio = float(round(max_c / max(min_c, 1), 2))
            is_imbalanced = class_imbalance_ratio >= 2.0

    # Sequence validation statistics
    sequence_stats: Dict[str, Any] = {}
    invalid_sequences_count = 0
    ambiguous_sequences_count = 0
    invalid_records_indices: List[int] = []

    if "sequence" in df.columns:
        sequence_stats = compute_sequence_stats(df["sequence"])

        for idx, seq in df["sequence"].items():
            valid, _, details = validate_sequence(
                seq,
                allowed_bases=allowed_bases,
                allow_ambiguous=allow_ambiguous,
                min_length=min_length,
                max_length=max_length,
            )
            if not valid:
                invalid_sequences_count += 1
                invalid_records_indices.append(int(idx))
            if details.get("ambiguous_characters"):
                ambiguous_sequences_count += 1

    validation_passed = (len(missing_required) == 0) and (total_records > 0)

    return {
        "validation_passed": validation_passed,
        "total_records": total_records,
        "total_columns": total_columns,
        "columns_present": columns_present,
        "missing_required_columns": missing_required,
        "available_supported_columns": available_supported,
        "unsupported_columns": unsupported_columns,
        "missing_values_per_column": missing_values,
        "exact_duplicates": exact_duplicates,
        "sequence_duplicates": sequence_duplicates,
        "class_distribution": class_distribution,
        "class_percentages": class_percentages,
        "class_imbalance_ratio": class_imbalance_ratio,
        "is_imbalanced": is_imbalanced,
        "sequence_stats": sequence_stats,
        "invalid_sequences_count": invalid_sequences_count,
        "ambiguous_sequences_count": ambiguous_sequences_count,
        "invalid_sample_indices": invalid_records_indices[:20],
    }

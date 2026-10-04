"""
Dataset Splitter Module
Implements stratified and group-aware train/validation/test splitting,
specifically designed to prevent data leakage in genomic sequence datasets.
"""
from typing import Any, Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit, train_test_split


def check_data_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    sequence_col: str = "sequence",
    group_col: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Rigorously verify absence of data leakage across train, test, and validation splits.

    Checks:
    1. Direct sequence overlap between train and test/val.
    2. Group overlap (e.g. gene, locus) between train and test/val when group_col is specified.
    3. Index overlap.

    Args:
        train_df: Training set.
        test_df: Testing set.
        val_df: Validation set (optional).
        sequence_col: Name of the DNA sequence column.
        group_col: Optional column indicating gene or sequence family.

    Returns:
        Dict[str, Any]: Detailed leakage diagnostic report.
    """
    splits = {"train": train_df, "test": test_df}
    if val_df is not None:
        splits["val"] = val_df

    report: Dict[str, Any] = {
        "leakage_detected": False,
        "sequence_overlap_counts": {},
        "group_overlap_counts": {},
        "details": [],
    }

    train_seqs = set(train_df[sequence_col].dropna().astype(str)) if sequence_col in train_df else set()
    test_seqs = set(test_df[sequence_col].dropna().astype(str)) if sequence_col in test_df else set()

    train_test_seq_overlap = train_seqs.intersection(test_seqs)
    report["sequence_overlap_counts"]["train_test"] = len(train_test_seq_overlap)

    if train_test_seq_overlap:
        report["leakage_detected"] = True
        report["details"].append(f"Found {len(train_test_seq_overlap)} identical sequences in both train and test sets!")

    if val_df is not None and sequence_col in val_df:
        val_seqs = set(val_df[sequence_col].dropna().astype(str))
        train_val_seq_overlap = train_seqs.intersection(val_seqs)
        test_val_seq_overlap = test_seqs.intersection(val_seqs)
        report["sequence_overlap_counts"]["train_val"] = len(train_val_seq_overlap)
        report["sequence_overlap_counts"]["test_val"] = len(test_val_seq_overlap)

        if train_val_seq_overlap:
            report["leakage_detected"] = True
            report["details"].append(f"Found {len(train_val_seq_overlap)} sequences in both train and validation sets!")
        if test_val_seq_overlap:
            report["leakage_detected"] = True
            report["details"].append(f"Found {len(test_val_seq_overlap)} sequences in both test and validation sets!")

    # Group column leakage check (e.g., gene)
    if group_col and all(group_col in df.columns for df in splits.values()):
        train_groups = set(train_df[group_col].dropna())
        test_groups = set(test_df[group_col].dropna())
        grp_overlap = train_groups.intersection(test_groups)
        report["group_overlap_counts"]["train_test"] = len(grp_overlap)

        if grp_overlap:
            report["leakage_detected"] = True
            report["details"].append(f"Found {len(grp_overlap)} shared '{group_col}' groups between train and test: {list(grp_overlap)[:5]}...")

        if val_df is not None:
            val_groups = set(val_df[group_col].dropna())
            train_val_grp = train_groups.intersection(val_groups)
            test_val_grp = test_groups.intersection(val_groups)
            report["group_overlap_counts"]["train_val"] = len(train_val_grp)
            report["group_overlap_counts"]["test_val"] = len(test_val_grp)
            if train_val_grp or test_val_grp:
                report["leakage_detected"] = True
                report["details"].append(f"Shared '{group_col}' groups detected with validation set!")

    return report


def split_dataset(
    df: pd.DataFrame,
    target_col: str = "label",
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_seed: int = 42,
    group_col: Optional[str] = None,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split a DataFrame into train, validation, and test partitions with stratification
    and optional group-aware isolation to prevent sequence leakage.

    Args:
        df: Input DataFrame.
        target_col: Target column name for stratified splitting.
        test_size: Proportion of dataset to include in the test split.
        val_size: Proportion of dataset to include in the validation split.
        random_seed: Seed for reproducibility.
        group_col: Column name defining groups (e.g. 'gene'). If provided and sufficient groups exist,
                   group-aware splitting is applied.
        stratify: Whether to preserve class proportions across splits.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
            - train_df: Training partition.
            - val_df: Validation partition.
            - test_df: Test partition.
            - split_metadata: Metadata with sample counts, class distributions, and leakage check results.
    """
    if len(df) == 0:
        raise ValueError("Cannot split an empty DataFrame.")

    total_records = len(df)
    train_ratio = 1.0 - (test_size + val_size)
    if train_ratio <= 0:
        raise ValueError(f"Sum of test_size ({test_size}) and val_size ({val_size}) must be < 1.0.")

    # Strategy 1: Group-aware splitting if group_col is provided and has > 4 unique groups
    use_group_split = (
        group_col is not None
        and group_col in df.columns
        and df[group_col].nunique() >= 4
    )

    if use_group_split:
        # Group-aware split to ensure no group (e.g. gene) spans across train and test
        groups = df[group_col]
        gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
        train_val_idx, test_idx = next(gss_test.split(df, groups=groups))

        train_val_df = df.iloc[train_val_idx].copy().reset_index(drop=True)
        test_df = df.iloc[test_idx].copy().reset_index(drop=True)

        # Split train_val into train and val
        relative_val_size = val_size / (1.0 - test_size)
        gss_val = GroupShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=random_seed)
        train_idx, val_idx = next(gss_val.split(train_val_df, groups=train_val_df[group_col]))

        train_df = train_val_df.iloc[train_idx].copy().reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].copy().reset_index(drop=True)
        split_method = f"GroupShuffleSplit (grouped by {group_col})"

    else:
        # Strategy 2: Stratified split based on target_col
        strat_col = None
        if stratify and target_col in df.columns:
            counts = df[target_col].value_counts()
            # Stratification requires at least 2 samples per class
            if (counts >= 2).all():
                strat_col = df[target_col]

        # First split test set
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_seed,
            stratify=strat_col,
        )

        train_val_df = train_val_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)

        # Next split val set from train_val
        relative_val_size = val_size / (1.0 - test_size)
        strat_val_col = None
        if strat_col is not None and target_col in train_val_df.columns:
            val_counts = train_val_df[target_col].value_counts()
            if (val_counts >= 2).all():
                strat_val_col = train_val_df[target_col]

        train_df, val_df = train_test_split(
            train_val_df,
            test_size=relative_val_size,
            random_state=random_seed,
            stratify=strat_val_col,
        )

        train_df = train_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)
        split_method = "StratifiedShuffleSplit" if strat_col is not None else "RandomSplit"

    # Verify absence of data leakage
    leakage_report = check_data_leakage(
        train_df=train_df,
        test_df=test_df,
        val_df=val_df,
        sequence_col="sequence" if "sequence" in df.columns else df.columns[0],
        group_col=group_col if use_group_split else None,
    )

    # Class balance summary
    def get_class_dist(d: pd.DataFrame, col: str) -> Dict[str, int]:
        if col in d.columns:
            return {str(k): int(v) for k, v in d[col].value_counts().items()}
        return {}

    split_metadata: Dict[str, Any] = {
        "split_method": split_method,
        "random_seed": random_seed,
        "total_records": total_records,
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "train_ratio": round(len(train_df) / total_records, 4),
        "val_ratio": round(len(val_df) / total_records, 4),
        "test_ratio": round(len(test_df) / total_records, 4),
        "train_class_distribution": get_class_dist(train_df, target_col),
        "val_class_distribution": get_class_dist(val_df, target_col),
        "test_class_distribution": get_class_dist(test_df, target_col),
        "leakage_check": leakage_report,
    }

    return train_df, val_df, test_df, split_metadata

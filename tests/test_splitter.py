"""
Unit tests for train/validation/test splitting and data leakage auditing.
"""
import pandas as pd
import pytest
from src.preprocessing.splitter import check_data_leakage, split_dataset


class TestSplitter:
    def test_stratified_split(self):
        # 100 samples with 50/50 balance
        seqs = [f"ATGCATGCATGC_{i}" for i in range(100)]
        labels = [0] * 50 + [1] * 50
        df = pd.DataFrame({"sequence": seqs, "label": labels})

        train_df, val_df, test_df, meta = split_dataset(
            df,
            target_col="label",
            test_size=0.2,
            val_size=0.1,
            random_seed=42,
            stratify=True,
        )

        assert len(train_df) == 70
        assert len(val_df) == 10
        assert len(test_df) == 20
        assert meta["train_records"] + meta["val_records"] + meta["test_records"] == 100

        # Check stratification balance
        train_ratio_0 = (train_df["label"] == 0).sum() / len(train_df)
        assert abs(train_ratio_0 - 0.5) < 0.05

        # Check data leakage audit
        leakage = check_data_leakage(train_df, test_df, val_df, sequence_col="sequence")
        assert leakage["leakage_detected"] is False

    def test_group_aware_split_prevents_leakage(self):
        # Multiple records belonging to distinct genes
        genes = ["TP53", "BRCA1", "EGFR", "BRAF", "KRAS", "CFTR", "APC", "PTEN"]
        rows = []
        for g in genes:
            for i in range(15):
                rows.append({
                    "sequence": f"SEQ_{g}_{i}",
                    "label": i % 2,
                    "gene": g,
                })
        df = pd.DataFrame(rows)

        train_df, val_df, test_df, meta = split_dataset(
            df,
            target_col="label",
            test_size=0.25,
            val_size=0.25,
            random_seed=42,
            group_col="gene",
        )

        # Check that no gene in train is in test or val
        train_genes = set(train_df["gene"])
        test_genes = set(test_df["gene"])
        val_genes = set(val_df["gene"])

        assert len(train_genes.intersection(test_genes)) == 0
        assert len(train_genes.intersection(val_genes)) == 0
        assert len(test_genes.intersection(val_genes)) == 0

        # Leakage check should report False
        leakage = meta["leakage_check"]
        assert leakage["leakage_detected"] is False

    def test_leakage_detection_triggers_on_overlap(self):
        # Intentionally overlapping datasets
        train_df = pd.DataFrame({"sequence": ["AAAA", "CCCC", "GGGG"], "label": [0, 1, 0]})
        test_df = pd.DataFrame({"sequence": ["CCCC", "TTTT"], "label": [1, 0]})

        leakage = check_data_leakage(train_df, test_df, sequence_col="sequence")
        assert leakage["leakage_detected"] is True
        assert leakage["sequence_overlap_counts"]["train_test"] == 1

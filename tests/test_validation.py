"""
Unit tests for DNA sequence and DataFrame validation.
"""
import pandas as pd
import pytest
from src.preprocessing.validator import (
    compute_sequence_stats,
    validate_dataframe,
    validate_sequence,
)


class TestSequenceValidation:
    def test_valid_sequence(self):
        is_valid, errors, details = validate_sequence("ATGCGATCGATC", min_length=5, max_length=100)
        assert is_valid is True
        assert len(errors) == 0
        assert details["length"] == 12
        assert len(details["invalid_characters"]) == 0

    def test_lowercase_conversion(self):
        # Even with lowercase input, validate_sequence handles uppercase conversion internally
        is_valid, errors, details = validate_sequence("atgcgatc", min_length=5)
        assert is_valid is True
        assert details["length"] == 8

    def test_invalid_characters(self):
        is_valid, errors, details = validate_sequence("ATGCZ9TAGC")
        assert is_valid is False
        assert len(errors) > 0
        assert "Z" in details["invalid_characters"]
        assert "9" in details["invalid_characters"]
        assert details["invalid_characters"]["Z"] == [4]
        assert details["invalid_characters"]["9"] == [5]

    def test_iupac_ambiguity_flagged(self):
        # By default allow_ambiguous is False
        is_valid, errors, details = validate_sequence("ATGCRNAT", allow_ambiguous=False)
        assert is_valid is False
        assert "R" in details["ambiguous_characters"]
        assert "N" in details["ambiguous_characters"]

        # When allow_ambiguous is True, it should pass
        is_valid_allowed, errors_allowed, _ = validate_sequence("ATGCRNAT", allow_ambiguous=True)
        assert is_valid_allowed is True
        assert len(errors_allowed) == 0

    def test_sequence_length_bounds(self):
        # Below min_length
        valid_short, errs_short, _ = validate_sequence("ATGC", min_length=10)
        assert valid_short is False
        assert any("below minimum" in e for e in errs_short)

        # Above max_length
        valid_long, errs_long, _ = validate_sequence("A" * 15, max_length=10)
        assert valid_long is False
        assert any("exceeds maximum" in e for e in errs_long)

    def test_empty_and_null_sequences(self):
        valid_empty, errs_empty, _ = validate_sequence("")
        assert valid_empty is False

        valid_none, errs_none, _ = validate_sequence(None)
        assert valid_none is False


class TestDataFrameValidation:
    def test_dataframe_inspection_complete(self):
        df = pd.DataFrame({
            "sequence": ["ATGCATGCATGC", "CGATCGATCGAT", "ATGCATGCATGC"],
            "label": [0, 1, 0],
            "gene": ["TP53", "BRCA1", "TP53"],
            "chromosome": ["chr17", "chr17", "chr17"],
        })

        summary = validate_dataframe(df, required_columns=["sequence", "label"])
        assert summary["validation_passed"] is True
        assert summary["total_records"] == 3
        assert summary["total_columns"] == 4
        assert "sequence" in summary["columns_present"]
        assert "label" in summary["columns_present"]
        assert "gene" in summary["available_supported_columns"]
        assert summary["exact_duplicates"] == 1  # 1st and 3rd rows are identical
        assert summary["sequence_duplicates"] == 1
        assert summary["class_distribution"] == {"0": 2, "1": 1}

    def test_dataframe_missing_required_column(self):
        df = pd.DataFrame({"sequence": ["ATGCATGCATGC"]})
        summary = validate_dataframe(df, required_columns=["sequence", "label"])
        assert summary["validation_passed"] is False
        assert "label" in summary["missing_required_columns"]

    def test_compute_sequence_stats(self):
        seqs = pd.Series(["ATGC", "ATGCATGC", "ATGCATGCATGC"])
        stats = compute_sequence_stats(seqs)
        assert stats["total_count"] == 3
        assert stats["min_length"] == 4
        assert stats["max_length"] == 12
        assert stats["mean_length"] == 8.0
        assert stats["median_length"] == 8.0

"""
Unit tests for DNA sequence cleaner and sanitizer.
"""
import pandas as pd
import pytest
from src.preprocessing.cleaner import DNADataCleaner, clean_sequence, handle_ambiguous_bases


class TestCleanSequence:
    def test_clean_sequence_formatting(self):
        raw = "  atg c\tgta\ncgtt  "
        cleaned = clean_sequence(raw, uppercase=True, strip_whitespace=True)
        assert cleaned == "ATGCGTACGTT"

    def test_clean_empty(self):
        assert clean_sequence(None) == ""
        assert clean_sequence("") == ""


class TestAmbiguousBases:
    def test_flag_ambiguity(self):
        seq, has_ambig = handle_ambiguous_bases("ATGCRNAT", action="flag")
        assert seq == "ATGCRNAT"
        assert has_ambig is True

    def test_replace_n(self):
        seq, has_ambig = handle_ambiguous_bases("ATGCRNAT", action="replace_n")
        assert seq == "ATGCNNAT"
        assert has_ambig is True

    def test_no_ambiguity(self):
        seq, has_ambig = handle_ambiguous_bases("ATGCATGC", action="replace_n")
        assert seq == "ATGCATGC"
        assert has_ambig is False


class TestDNADataCleaner:
    def test_clean_dataset_filters_and_encodes(self):
        raw_df = pd.DataFrame({
            "sequence": [
                "atgcatgcatgc",        # Valid lowercase (12 bp)
                "ATGCATGCATGC",        # Duplicate sequence (12 bp)
                "ATG C\tGCA\nTGCA",    # Sequence with whitespace (12 bp)
                "ATGCZ9XTAGC",         # Invalid characters
                "ATGC",                # Too short (< 10 bp)
                None,                  # Missing sequence
                "CGATCGATCGAT",        # Valid (12 bp)
            ],
            "label": ["wildtype", "wildtype", "mutant", "mutant", "mutant", "mutant", "mutant"],
        })

        cleaner = DNADataCleaner(
            min_length=10,
            max_length=50,
            remove_duplicates=True,
            ambiguous_action="filter",
        )

        cleaned_df, audit = cleaner.clean_dataset(raw_df)

        assert audit["initial_records"] == 7
        assert audit["dropped_missing_sequence"] == 1
        assert audit["dropped_invalid_nucleotides"] == 1
        assert audit["dropped_length_constraints"] == 1
        assert audit["dropped_duplicates"] >= 1

        # Check label encoding
        assert "label_encoded" in cleaned_df.columns
        assert set(cleaned_df["label_encoded"].unique()).issubset({0, 1})
        assert "wildtype" in cleaner.label_mapping
        assert "mutant" in cleaner.label_mapping

        # All sequences in cleaned_df must be uppercase and without whitespace
        for seq in cleaned_df["sequence"]:
            assert seq.isupper()
            assert " " not in seq
            assert len(seq) >= 10

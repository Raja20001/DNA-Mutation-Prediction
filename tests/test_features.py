"""
Unit tests for DNA feature engineering module.
"""
import pandas as pd
import pytest
from src.features.extractor import (
    DNAFeatureExtractor,
    compute_basic_composition,
    compute_kmer_frequencies,
    compute_shannon_entropy,
    extract_mutation_context_features,
)


class TestFeatureExtraction:
    def test_basic_composition(self):
        seq = "ATGC"
        comp = compute_basic_composition(seq)
        assert comp["seq_length"] == 4.0
        assert comp["freq_A"] == 0.25
        assert comp["freq_C"] == 0.25
        assert comp["freq_G"] == 0.25
        assert comp["freq_T"] == 0.25
        assert comp["gc_content"] == 0.5
        assert comp["at_content"] == 0.5

    def test_kmer_frequencies_sum_to_one(self):
        seq = "ATGCATGCATGCATGC"
        kmers_2 = compute_kmer_frequencies(seq, k=2)
        assert len(kmers_2) == 16  # 4^2 = 16
        total_freq = sum(kmers_2.values())
        assert abs(total_freq - 1.0) < 1e-4

        kmers_3 = compute_kmer_frequencies(seq, k=3)
        assert len(kmers_3) == 64  # 4^3 = 64

    def test_shannon_entropy_bounds(self):
        # Homopolymer: minimum entropy = 0.0
        homopolymer = "AAAAAAAAAAAA"
        assert compute_shannon_entropy(homopolymer) == 0.0

        # Equal distribution of 4 bases: maximum entropy = 2.0
        balanced = "ACGTACGTACGT"
        assert compute_shannon_entropy(balanced) == 2.0

    def test_mutation_context(self):
        ref = "ATGCCATGGA"
        alt = "ATGCTATGGA"
        ctx = extract_mutation_context_features(ref, alt, window_size=2)
        assert ctx["has_mutation_context"] == 1
        assert ctx["mut_pos_relative"] == 5
        assert ctx["mut_ref_base"] == "C"
        assert ctx["mut_alt_base"] == "T"

    def test_feature_extractor_class(self):
        df_train = pd.DataFrame({"sequence": ["ATGCATGCATGC", "CGATCGATCGAT", "GGGAAATTCCCC"]})
        df_test = pd.DataFrame({"sequence": ["ATGCATGCATGC"]})

        extractor = DNAFeatureExtractor(kmer_sizes=[2], scale_features=True)
        extractor.fit(df_train)

        X_train = extractor.transform(df_train, as_dataframe=True)
        X_test = extractor.transform(df_test, as_dataframe=True)

        assert len(X_train) == 3
        assert len(X_test) == 1
        assert X_train.shape[1] == X_test.shape[1]
        assert "gc_content" in X_train.columns
        assert "kmer_2_AA" in X_train.columns

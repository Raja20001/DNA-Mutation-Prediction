"""
DNA Feature Engineering Module
Extracts:
1. Basic Composition (A, C, G, T frequencies, GC content, AT content, GC/AT ratio)
2. K-mer Spectrum (1-mer, 2-mer, 3-mer, configurable k-mers)
3. Information Entropy (Shannon entropy) & Nucleotide Diversity (Gini-Simpson index)
4. Context Features (CpG density, GC skew, maximum homopolymer run, sequence complexity)
5. Optional Reference-Aware RKNP features (Reference K-mer Novelty Profile)
"""
from itertools import product
import math
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .rknp import compute_rknp_features

NUCLEOTIDES = ["A", "C", "G", "T"]


def generate_all_kmers(k: int) -> List[str]:
    """Generate all 4^k possible canonical k-mers in lexicographical order."""
    return ["".join(p) for p in product(NUCLEOTIDES, repeat=k)]


def compute_basic_composition(sequence: str) -> Dict[str, float]:
    """
    Compute basic sequence length, nucleotide frequencies, GC content, and AT content.
    """
    seq_len = len(sequence)
    if seq_len == 0:
        return {
            "seq_length": 0.0,
            "freq_A": 0.0,
            "freq_C": 0.0,
            "freq_G": 0.0,
            "freq_T": 0.0,
            "gc_content": 0.0,
            "at_content": 0.0,
            "gc_at_ratio": 0.0,
        }

    count_a = sequence.count("A")
    count_c = sequence.count("C")
    count_g = sequence.count("G")
    count_t = sequence.count("T")

    freq_a = count_a / seq_len
    freq_c = count_c / seq_len
    freq_g = count_g / seq_len
    freq_t = count_t / seq_len

    gc_content = (count_g + count_c) / seq_len
    at_content = (count_a + count_t) / seq_len
    gc_at_ratio = (count_g + count_c) / max(count_a + count_t, 1)

    return {
        "seq_length": float(seq_len),
        "freq_A": float(round(freq_a, 4)),
        "freq_C": float(round(freq_c, 4)),
        "freq_G": float(round(freq_g, 4)),
        "freq_T": float(round(freq_t, 4)),
        "gc_content": float(round(gc_content, 4)),
        "at_content": float(round(at_content, 4)),
        "gc_at_ratio": float(round(gc_at_ratio, 4)),
    }


def compute_kmer_frequencies(sequence: str, k: int = 2) -> Dict[str, float]:
    """
    Compute normalized frequencies for all 4^k k-mers using a sliding window of step 1.
    """
    all_kmers = generate_all_kmers(k)
    kmer_counts = {f"kmer_{k}_{km}": 0 for km in all_kmers}

    seq_len = len(sequence)
    total_kmers = seq_len - k + 1

    if total_kmers <= 0:
        return {f"kmer_{k}_{km}": 0.0 for km in all_kmers}

    for i in range(total_kmers):
        km = sequence[i : i + k]
        feat_name = f"kmer_{k}_{km}"
        if feat_name in kmer_counts:
            kmer_counts[feat_name] += 1

    return {kmer: float(round(count / total_kmers, 5)) for kmer, count in kmer_counts.items()}


def compute_shannon_entropy(sequence: str) -> float:
    """
    Compute Shannon information entropy H = - sum(p_i * log2(p_i)) of nucleotide distribution.
    Reflects sequence complexity (maximum value is 2.0 bits for 4 uniform bases).
    """
    seq_len = len(sequence)
    if seq_len == 0:
        return 0.0

    entropy = 0.0
    for base in NUCLEOTIDES:
        p = sequence.count(base) / seq_len
        if p > 0:
            entropy -= p * math.log2(p)

    return float(round(entropy, 4))


def compute_nucleotide_diversity(sequence: str) -> float:
    """
    Compute Gini-Simpson nucleotide diversity index: 1 - sum(p_i^2).
    """
    seq_len = len(sequence)
    if seq_len == 0:
        return 0.0

    sum_sq = sum((sequence.count(base) / seq_len) ** 2 for base in NUCLEOTIDES)
    return float(round(1.0 - sum_sq, 4))


def compute_context_features(sequence: str) -> Dict[str, float]:
    """
    Compute contextual genomic features:
    - CpG island density & observed/expected CpG ratio
    - GC skew: (G - C) / (G + C)
    - AT skew: (A - T) / (A + T)
    - Max homopolymer run length
    """
    seq_len = len(sequence)
    if seq_len == 0:
        return {
            "cpg_density": 0.0,
            "cpg_oe_ratio": 0.0,
            "gc_skew": 0.0,
            "at_skew": 0.0,
            "max_homopolymer": 0.0,
        }

    c_cnt = sequence.count("C")
    g_cnt = sequence.count("G")
    a_cnt = sequence.count("A")
    t_cnt = sequence.count("T")

    # CpG count
    cpg_cnt = sequence.count("CG")
    cpg_density = cpg_cnt / max(1, seq_len - 1)
    expected_cpg = (c_cnt * g_cnt) / max(1, seq_len)
    cpg_oe = (cpg_cnt / expected_cpg) if expected_cpg > 0 else 0.0

    # Skews
    gc_skew = (g_cnt - c_cnt) / max(1, g_cnt + c_cnt)
    at_skew = (a_cnt - t_cnt) / max(1, a_cnt + t_cnt)

    # Longest homopolymer run
    max_run = 1
    curr_run = 1
    for i in range(1, seq_len):
        if sequence[i] == sequence[i - 1]:
            curr_run += 1
            if curr_run > max_run:
                max_run = curr_run
        else:
            curr_run = 1

    return {
        "cpg_density": float(round(cpg_density, 5)),
        "cpg_oe_ratio": float(round(cpg_oe, 4)),
        "gc_skew": float(round(gc_skew, 4)),
        "at_skew": float(round(at_skew, 4)),
        "max_homopolymer": float(max_run),
    }


def extract_mutation_context_features(
    ref_seq: str,
    alt_seq: str,
    window_size: int = 5,
) -> Dict[str, Any]:
    """
    Extract mutation context features when both reference and alternate sequences are available.
    """
    min_len = min(len(ref_seq), len(alt_seq))
    diff_pos = None

    for i in range(min_len):
        if ref_seq[i] != alt_seq[i]:
            diff_pos = i
            break

    if diff_pos is None:
        if len(ref_seq) != len(alt_seq):
            diff_pos = min_len
        else:
            return {
                "mut_pos_relative": 0,
                "mut_ref_base": "None",
                "mut_alt_base": "None",
                "flanking_gc": 0.0,
                "has_mutation_context": 0,
            }

    ref_base = ref_seq[diff_pos] if diff_pos < len(ref_seq) else "-"
    alt_base = alt_seq[diff_pos] if diff_pos < len(alt_seq) else "-"

    start_win = max(0, diff_pos - window_size)
    end_win = min(len(ref_seq), diff_pos + window_size + 1)
    flank_seq = ref_seq[start_win:end_win]

    flank_gc = (flank_seq.count("G") + flank_seq.count("C")) / max(len(flank_seq), 1)

    return {
        "mut_pos_relative": diff_pos + 1,
        "mut_ref_base": ref_base,
        "mut_alt_base": alt_base,
        "flanking_gc": float(round(flank_gc, 4)),
        "has_mutation_context": 1,
    }


class DNAFeatureExtractor:
    """
    Extracts tabular feature representations from DNA sequences.
    Combines:
    - Basic composition (nucleotide frequencies, GC, AT, length)
    - Configurable k-mer spectrums (e.g. k=2, 3)
    - Information entropy & nucleotide diversity
    - Optional sequence context (CpG, skews, homopolymer)
    - Optional Reference-Aware RKNP features
    Maintains fitted scalers for reproducible, leakage-free inference.
    """

    def __init__(
        self,
        kmer_sizes: Optional[List[int]] = None,
        include_entropy: bool = True,
        include_diversity: bool = True,
        include_context: bool = False,
        include_rknp: bool = False,
        scale_features: bool = True,
        scaler_type: str = "standard",
    ):
        self.kmer_sizes = kmer_sizes if kmer_sizes is not None else [2, 3]
        self.include_entropy = include_entropy
        self.include_diversity = include_diversity
        self.include_context = include_context
        self.include_rknp = include_rknp
        self.scale_features = scale_features
        self.scaler_type = scaler_type

        self.feature_names_: List[str] = []
        self.scaler_: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.is_fitted_: bool = False

    def extract_from_sequence(self, sequence: str, reference_seq: Optional[str] = None) -> Dict[str, float]:
        """Extract all configured numerical features from a single DNA sequence string."""
        features: Dict[str, float] = {}
        clean_seq = "".join(sequence.split()).upper()

        # 1. Basic composition
        features.update(compute_basic_composition(clean_seq))

        # 2. Information entropy & diversity
        if self.include_entropy:
            features["shannon_entropy"] = compute_shannon_entropy(clean_seq)
        if self.include_diversity:
            features["nucleotide_diversity"] = compute_nucleotide_diversity(clean_seq)

        # 3. Context features
        if self.include_context:
            features.update(compute_context_features(clean_seq))

        # 4. K-mer frequencies
        for k in self.kmer_sizes:
            features.update(compute_kmer_frequencies(clean_seq, k=k))

        # 5. Reference K-mer Novelty Profile (RKNP)
        if self.include_rknp:
            features.update(compute_rknp_features(clean_seq, reference_seq=reference_seq))

        return features

    def fit(self, df: pd.DataFrame, sequence_col: str = "sequence", reference_col: Optional[str] = None) -> "DNAFeatureExtractor":
        """
        Fit the feature extractor and internal scaler on training DataFrame.
        """
        raw_features = []
        for _, row in df.iterrows():
            seq = row[sequence_col]
            ref = row[reference_col] if (reference_col and reference_col in df.columns) else None
            raw_features.append(self.extract_from_sequence(seq, reference_seq=ref))

        feature_df = pd.DataFrame(raw_features)
        self.feature_names_ = list(feature_df.columns)

        if self.scale_features:
            if self.scaler_type == "minmax":
                self.scaler_ = MinMaxScaler()
            else:
                self.scaler_ = StandardScaler()
            self.scaler_.fit(feature_df.values)

        self.is_fitted_ = True
        return self

    def transform(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        reference_col: Optional[str] = None,
        as_dataframe: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """
        Transform a DataFrame of DNA sequences into the numerical feature matrix.
        """
        if not self.is_fitted_:
            raise RuntimeError("DNAFeatureExtractor must be fitted before calling transform().")

        raw_features = []
        for _, row in df.iterrows():
            seq = row[sequence_col]
            ref = row[reference_col] if (reference_col and reference_col in df.columns) else None
            raw_features.append(self.extract_from_sequence(seq, reference_seq=ref))

        feature_df = pd.DataFrame(raw_features, columns=self.feature_names_)

        if self.scale_features and self.scaler_ is not None:
            scaled_vals = self.scaler_.transform(feature_df.values)
            if as_dataframe:
                return pd.DataFrame(scaled_vals, columns=self.feature_names_, index=df.index)
            return scaled_vals

        if as_dataframe:
            return feature_df
        return feature_df.values

    def fit_transform(
        self,
        df: pd.DataFrame,
        sequence_col: str = "sequence",
        reference_col: Optional[str] = None,
        as_dataframe: bool = True,
    ) -> Union[pd.DataFrame, np.ndarray]:
        """Fit on DataFrame and return transformed feature matrix."""
        return self.fit(df, sequence_col=sequence_col, reference_col=reference_col).transform(
            df, sequence_col=sequence_col, reference_col=reference_col, as_dataframe=as_dataframe
        )

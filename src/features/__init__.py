"""
DNA Feature Engineering Package.
"""
from .extractor import (
    DNAFeatureExtractor,
    compute_basic_composition,
    compute_context_features,
    compute_kmer_frequencies,
    compute_nucleotide_diversity,
    compute_shannon_entropy,
    extract_mutation_context_features,
    generate_all_kmers,
)
from .pipeline import run_feature_pipeline
from .rknp import compute_rknp_features, simulate_noisy_reference

__all__ = [
    "DNAFeatureExtractor",
    "compute_basic_composition",
    "compute_context_features",
    "compute_kmer_frequencies",
    "compute_shannon_entropy",
    "compute_nucleotide_diversity",
    "extract_mutation_context_features",
    "generate_all_kmers",
    "run_feature_pipeline",
    "compute_rknp_features",
    "simulate_noisy_reference",
]

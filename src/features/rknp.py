"""
Reference K-mer Novelty Profile (RKNP) & Reference-Conditioned Feature Module
Calculates:
- novel_kmer_count: Absolute count of sample k-mers absent in reference sequence
- novelty_fraction: Fraction of sample k-mers that are novel relative to reference
- longest_novelty_run: Longest contiguous run of novel k-mers
- reference_similarity: Jaccard & cosine similarity between reference and sample k-mer spectra
- reference_conditioned_score: Calibrated composite metric of mutation perturbation
Provides explicit reference-blind default fallback when reference sequence is unavailable.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np


def extract_kmers_set(sequence: str, k: int) -> Set[str]:
    """Extract set of unique canonical k-mers from sequence."""
    seq = "".join(sequence.split()).upper()
    if len(seq) < k:
        return set()
    return {seq[i : i + k] for i in range(len(seq) - k + 1)}


def extract_kmers_list(sequence: str, k: int) -> List[str]:
    """Extract ordered list of sliding k-mers."""
    seq = "".join(sequence.split()).upper()
    if len(seq) < k:
        return []
    return [seq[i : i + k] for i in range(len(seq) - k + 1)]


def compute_rknp_features(
    sample_seq: str,
    reference_seq: Optional[str] = None,
    k_values: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute Reference K-mer Novelty Profile (RKNP) metrics.
    If reference_seq is None or empty, returns REFERENCE-BLIND zeroed/neutral defaults.
    """
    if k_values is None:
        k_values = [3, 4]

    features: Dict[str, float] = {}

    clean_sample = "".join(str(sample_seq).split()).upper()
    has_ref = bool(reference_seq and str(reference_seq).strip())

    if not has_ref:
        # REFERENCE-BLIND MODE: Neutral fallback values
        features["reference_available"] = 0.0
        features["ref_similarity_jaccard"] = 0.50
        features["ref_similarity_containment"] = 0.50
        features["rknp_conditioned_score"] = 0.0

        for k in k_values:
            features[f"rknp_k{k}_novel_count"] = 0.0
            features[f"rknp_k{k}_novelty_fraction"] = 0.0
            features[f"rknp_k{k}_longest_run"] = 0.0
        return features

    clean_ref = "".join(str(reference_seq).split()).upper()
    features["reference_available"] = 1.0

    total_novel_fractions = []

    for k in k_values:
        ref_kmers = extract_kmers_set(clean_ref, k)
        sample_kmers_ordered = extract_kmers_list(clean_sample, k)
        sample_kmers_set = set(sample_kmers_ordered)

        total_sample_kmers = len(sample_kmers_ordered)

        if total_sample_kmers == 0 or len(ref_kmers) == 0:
            features[f"rknp_k{k}_novel_count"] = 0.0
            features[f"rknp_k{k}_novelty_fraction"] = 0.0
            features[f"rknp_k{k}_longest_run"] = 0.0
            continue

        # Novel k-mers: k-mers present in sample but completely absent in reference
        novel_kmers = [km for km in sample_kmers_ordered if km not in ref_kmers]
        novel_count = len(set(novel_kmers))
        novelty_frac = len(novel_kmers) / max(total_sample_kmers, 1)

        # Longest contiguous run of novel k-mers in the query sequence
        longest_run = 0
        current_run = 0
        for km in sample_kmers_ordered:
            if km not in ref_kmers:
                current_run += 1
                if current_run > longest_run:
                    longest_run = current_run
            else:
                current_run = 0

        features[f"rknp_k{k}_novel_count"] = float(novel_count)
        features[f"rknp_k{k}_novelty_fraction"] = float(round(novelty_frac, 5))
        features[f"rknp_k{k}_longest_run"] = float(longest_run)
        total_novel_fractions.append(novelty_frac)

    # Multi-k Jaccard similarity (using k=3)
    k_base = k_values[0] if k_values else 3
    ref_set_base = extract_kmers_set(clean_ref, k_base)
    sample_set_base = extract_kmers_set(clean_sample, k_base)

    intersection = len(ref_set_base & sample_set_base)
    union = len(ref_set_base | sample_set_base)
    jaccard = (intersection / union) if union > 0 else 1.0
    containment = (intersection / len(sample_set_base)) if sample_set_base else 1.0

    features["ref_similarity_jaccard"] = float(round(jaccard, 5))
    features["ref_similarity_containment"] = float(round(containment, 5))

    # Reference-Conditioned Perturbation Score (0 = identical, 1 = completely divergent)
    avg_novelty = np.mean(total_novel_fractions) if total_novel_fractions else 0.0
    conditioned_score = float(round(0.6 * avg_novelty + 0.4 * (1.0 - jaccard), 5))
    features["rknp_conditioned_score"] = min(1.0, max(0.0, conditioned_score))

    return features


def simulate_noisy_reference(reference_seq: str, noise_rate: float = 0.05, seed: int = 42) -> str:
    """
    Simulate sequencing or assembly noise in reference sequence for robustness testing.
    """
    rng = np.random.RandomState(seed)
    bases = ["A", "C", "G", "T"]
    clean_ref = list("".join(reference_seq.split()).upper())

    for i in range(len(clean_ref)):
        if rng.uniform(0, 1) < noise_rate:
            # Substitute with a different random base
            current = clean_ref[i]
            alt_choices = [b for b in bases if b != current]
            clean_ref[i] = rng.choice(alt_choices)

    return "".join(clean_ref)

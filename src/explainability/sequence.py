"""
Sequence Explainability Module for TCN
Computes per-nucleotide saliency and gradient-based attribution scores along the DNA sequence.
Reveals which sequence regions and positional motifs drive model classifications.

CRITICAL RESEARCH RULE:
Sequence attributions are computational feature importances of the neural network,
not definitive biophysical contact points.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch

from ..deep_learning.tcn import NUCLEOTIDE_TO_IDX, encode_dna_sequences


def compute_tcn_sequence_saliency(
    model: torch.nn.Module,
    sequence: str,
    target_class: int = 1,
    max_len: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute gradient-based attribution saliency map for each nucleotide in a DNA sequence.

    Args:
        model: Trained TCNSequenceClassifier instance.
        sequence: Raw uppercase DNA sequence string.
        target_class: Target class index to differentiate with respect to (default: 1, mutation).
        max_len: Sequence length for tensor alignment (defaults to length of input sequence).

    Returns:
        Dict[str, Any] containing:
            - sequence: str
            - positions: List[int] (1-based indices)
            - nucleotides: List[str]
            - saliency_scores: List[float] (normalized in [0, 1])
            - top_attended_positions: List[Dict[str, Any]]
            - peak_saliency_position: int
    """
    model.eval()
    seq_str = str(sequence).strip().upper()
    seq_len = len(seq_str)
    pad_len = max_len or seq_len

    # Encode sequence to token indices
    encoded = encode_dna_sequences([seq_str], max_len=pad_len)
    input_tensor = torch.tensor(encoded, dtype=torch.long)

    # Forward pass through embedding layer with gradient tracking
    embedded = model.embedding(input_tensor)  # (1, pad_len, emb_dim)
    embedded.retain_grad()

    # Pass embedded tensor through remaining network
    emb_permuted = embedded.permute(0, 2, 1)  # (1, emb_dim, pad_len)
    features = model.tcn(emb_permuted)
    pooled = model.global_pool(features).squeeze(-1)
    logits = model.classifier(pooled)

    # Compute gradient of target class score with respect to embeddings
    target_score = logits[0, target_class] if logits.shape[1] > target_class else logits[0, 0]
    model.zero_grad()
    target_score.backward()

    # Extract gradients at the embedding level
    grads = embedded.grad.data.squeeze(0).numpy()  # (pad_len, emb_dim)
    # L2 norm across embedding dimension for each nucleotide position
    raw_saliency = np.linalg.norm(grads[:seq_len], axis=1)

    # Normalize to [0, 1] for comparative visualization
    max_val = np.max(raw_saliency) if len(raw_saliency) > 0 else 1.0
    norm_saliency = (raw_saliency / (max_val + 1e-8)).tolist()

    positions = list(range(1, seq_len + 1))
    nucleotides = list(seq_str)

    pos_records = [
        {"position": pos, "nucleotide": nuc, "saliency": round(score, 4)}
        for pos, nuc, score in zip(positions, nucleotides, norm_saliency)
    ]
    top_positions = sorted(pos_records, key=lambda x: x["saliency"], reverse=True)[:5]
    peak_pos = top_positions[0]["position"] if top_positions else 1

    return {
        "sequence": seq_str,
        "positions": positions,
        "nucleotides": nucleotides,
        "saliency_scores": [round(s, 4) for s in norm_saliency],
        "top_attended_positions": top_positions,
        "peak_saliency_position": peak_pos,
        "note": "Attribution indicates positions with highest gradient magnitude towards mutation prediction.",
    }


# Standard alias for unified master pipeline
compute_sequence_saliency = compute_tcn_sequence_saliency


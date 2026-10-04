"""
Explainability Package: Tabular, Sequence, and Quantum Attribution
"""
from .quantum_sensitivity import compute_quantum_feature_sensitivity
from .sequence import compute_sequence_saliency, compute_tcn_sequence_saliency
from .tabular import TabularExplainer
from .visualizer import plot_feature_importance, plot_sequence_saliency

__all__ = [
    "TabularExplainer",
    "compute_quantum_feature_sensitivity",
    "compute_sequence_saliency",
    "compute_tcn_sequence_saliency",
    "plot_feature_importance",
    "plot_sequence_saliency",
]


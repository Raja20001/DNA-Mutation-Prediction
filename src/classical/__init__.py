"""
Classical Machine Learning Baselines Package.
"""
from .evaluator import compute_classification_metrics
from .models import ClassicalBenchmark, build_classical_models

__all__ = ["ClassicalBenchmark", "build_classical_models", "compute_classification_metrics"]

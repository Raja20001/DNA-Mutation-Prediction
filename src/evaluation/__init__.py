"""
Evaluation Package: Model Comparison, Cross-Validation, and Statistical Validation
"""
from .comparator import ModelComparator
from .statistical_tests import compare_models_statistical_test, compute_confidence_interval

__all__ = [
    "ModelComparator",
    "compare_models_statistical_test",
    "compute_confidence_interval",
]

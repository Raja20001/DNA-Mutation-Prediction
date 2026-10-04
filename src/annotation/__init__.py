"""
Annotation Module: Normalization, Genomic Annotation, and Biological Interpretation
"""
from .annotator import GenomicAnnotator
from .biological_interpretation import interpret_variant
from .normalizer import NormalizedVariant, normalize_variant

__all__ = ["GenomicAnnotator", "NormalizedVariant", "interpret_variant", "normalize_variant"]

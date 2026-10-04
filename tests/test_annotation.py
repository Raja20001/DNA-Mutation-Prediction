"""
Unit Tests for Variant Normalization, Genomic Annotation, and Biological Interpretation
"""
import pandas as pd
import pytest

from src.annotation.annotator import GenomicAnnotator
from src.annotation.biological_interpretation import interpret_variant
from src.annotation.normalizer import normalize_variant


class TestAnnotation:
    def test_normalize_variant_provenance(self):
        row = {
            "chromosome": "7",
            "position": 140453190,
            "reference": "C",
            "alternate": "T",
            "gene": "BRAF",
            "transcript": "ENST00000288602",
        }
        var = normalize_variant(row)
        assert var.chromosome == "chr7"
        assert var.position == 140453190
        assert var.gene == "BRAF"
        assert "chromosome" in var.supplied_fields
        assert "position" in var.supplied_fields

    def test_normalize_variant_missing_fields(self):
        row = {"sequence": "ATGCATGC"}
        var = normalize_variant(row)
        assert var.chromosome == "chrUnknown"
        assert var.position is None
        assert "chromosome" in var.derived_fields

    def test_genomic_annotator(self):
        row = {
            "chromosome": "chr7",
            "position": 140453190,
            "reference": "A",
            "alternate": "T",
            "gene": "BRAF",
        }
        var = normalize_variant(row)
        annotator = GenomicAnnotator()
        annot = annotator.annotate(var)
        assert annot["gene"] == "BRAF"
        assert "Kinase domain" in annot["region"] or "Exon" in annot["region"]
        assert annot["molecular_consequence"] == "missense_variant"

    def test_biological_interpretation_separation(self):
        row = {
            "chromosome": "chr7",
            "position": 140453190,
            "reference": "A",
            "alternate": "T",
            "gene": "BRAF",
        }
        var = normalize_variant(row)
        pred = {"mutation_detected": "YES", "mutation_probability": 0.94}
        evidence = {"source": "ClinVar", "population_frequency": "0.0001"}

        interp = interpret_variant(var, prediction_result=pred, external_evidence=evidence)
        assert "model_prediction" in interp
        assert "database_evidence" in interp
        assert interp["model_prediction"]["mutation_detected"] == "YES"
        assert interp["database_evidence"]["gene_symbol"] == "BRAF"

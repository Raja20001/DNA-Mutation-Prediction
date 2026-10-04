"""
Unit Tests for Mutation Detection, Classification, and Localization
"""
import numpy as np
import pandas as pd
import pytest

from src.mutation.classifier import MutationClassifier
from src.mutation.detector import MutationDetector
from src.mutation.localizer import localize_mutation


class TestMutationAnalysis:
    def test_localization_substitution(self):
        ref = "ATGCCATGGA"
        alt = "ATGCTATGGA"
        res = localize_mutation(ref, alt)
        assert res["localized"] is True
        assert res["mutation_type"] == "substitution"
        assert res["reference_base"] == "C"
        assert res["alternate_base"] == "T"
        assert res["change"] == "C>T"
        assert res["local_position"] == 5

    def test_localization_insertion(self):
        ref = "ATGCAT"
        alt = "ATGCTAT"
        res = localize_mutation(ref, alt)
        assert res["localized"] is True
        assert res["mutation_type"] == "insertion"
        assert res["reference_base"] == "-"
        assert res["alternate_base"] == "T"

    def test_localization_duplication(self):
        ref = "ATGCAT"
        alt = "ATGCAAT"
        res = localize_mutation(ref, alt)
        assert res["localized"] is True
        assert res["mutation_type"] == "duplication"
        assert res["change"] == "dupA"

    def test_localization_deletion(self):
        ref = "ATGCAT"
        alt = "ATGAT"
        res = localize_mutation(ref, alt)
        assert res["localized"] is True
        assert res["mutation_type"] == "deletion"
        assert res["reference_base"] == "C"
        assert res["alternate_base"] == "-"

    def test_localization_identical_wildtype(self):
        ref = "ATGCAT"
        alt = "ATGCAT"
        res = localize_mutation(ref, alt)
        assert res["localized"] is True
        assert res["mutation_type"] == "wildtype"
        assert res["change"] == "None (Identical)"

    def test_localization_missing_data(self):
        res = localize_mutation(None, "ATGC")
        assert res["localized"] is False
        assert "unavailable" in res["message"].lower()

    def test_detector_pairwise(self):
        detector = MutationDetector()
        res_mut = detector.detect_from_sequences("ATGC", "ATTC")
        assert res_mut["mutation_detected"] == "YES"
        assert res_mut["detected_boolean"] is True

        res_wt = detector.detect_from_sequences("ATGC", "ATGC")
        assert res_wt["mutation_detected"] == "NO"
        assert res_wt["detected_boolean"] is False

    def test_classifier_deterministic(self):
        classifier = MutationClassifier()
        res = classifier.classify_from_sequences("A", "G")
        assert res["mutation_type"] == "Substitution"
        assert res["prediction_probability"] == 1.0

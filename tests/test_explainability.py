"""
Unit Tests for Explainable AI (Tabular, Sequence, and Quantum)
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

from src.deep_learning.tcn import TCNSequenceClassifier
from src.explainability.sequence import compute_tcn_sequence_saliency
from src.explainability.tabular import TabularExplainer


class TestExplainability:
    def test_tabular_explainer(self):
        np.random.seed(42)
        X = np.random.randn(20, 5)
        y = np.random.randint(0, 2, size=20)
        clf = RandomForestClassifier(n_estimators=10, random_state=42)
        clf.fit(X, y)

        feat_names = ["f1", "f2", "f3", "f4", "f5"]
        explainer = TabularExplainer(clf, feature_names=feat_names)

        df_imp = explainer.get_feature_importances(top_n=3)
        assert len(df_imp) == 3
        assert "feature" in df_imp.columns
        assert "importance" in df_imp.columns

        df_local = explainer.explain_sample(X[0], top_n=2)
        assert len(df_local) == 2
        assert "attribution_score" in df_local.columns

    def test_tcn_sequence_saliency(self):
        model = TCNSequenceClassifier(
            num_classes=2,
            embedding_dim=8,
            num_filters=16,
            kernel_size=3,
            num_levels=2,
        )
        seq = "ATGCCATGGA"
        saliency_dict = compute_tcn_sequence_saliency(model, seq)
        assert len(saliency_dict["saliency_scores"]) == len(seq)
        assert len(saliency_dict["positions"]) == len(seq)
        assert saliency_dict["peak_saliency_position"] in range(1, len(seq) + 1)

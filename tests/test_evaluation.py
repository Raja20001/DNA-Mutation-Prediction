"""
Unit Tests for Model Evaluation & Statistical Testing
"""
import numpy as np
import pytest
from src.evaluation.comparator import ModelComparator
from src.evaluation.statistical_tests import (
    compare_models_statistical_test,
    compute_confidence_interval,
)


class TestEvaluation:
    def test_model_comparator(self):
        comparator = ModelComparator()
        comparator.add_model_result("Random Forest", {"accuracy": 0.92, "f1_score": 0.91, "train_time_sec": 0.3})
        comparator.add_model_result("QMFN", {"accuracy": 0.95, "f1_score": 0.94, "train_time_sec": 1.2})

        df_table = comparator.get_comparison_table()
        assert len(df_table) == 2
        assert "Model" in df_table.columns
        assert "F1-Score" in df_table.columns

    def test_confidence_interval(self):
        scores = [0.90, 0.92, 0.91, 0.93, 0.89]
        ci = compute_confidence_interval(scores, confidence=0.95)
        assert ci["mean"] == pytest.approx(0.91, abs=0.01)
        assert ci["ci_lower"] <= ci["mean"] <= ci["ci_upper"]

    def test_paired_statistical_test(self):
        scores_a = [0.95, 0.96, 0.94, 0.97, 0.95]
        scores_b = [0.85, 0.87, 0.86, 0.88, 0.84]
        res = compare_models_statistical_test(
            scores_a, scores_b, model_a_name="Model A", model_b_name="Model B"
        )
        assert res["statistically_significant"] is True
        assert res["p_value"] < 0.05
        assert "Model A outperforms Model B" in res["interpretation"]

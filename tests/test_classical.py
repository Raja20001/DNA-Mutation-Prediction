"""
Unit tests for classical machine learning baselines.
"""
import numpy as np
import pandas as pd
import pytest
from src.classical.models import ClassicalBenchmark, build_classical_models


class TestClassicalModels:
    def test_model_instantiation(self):
        models = build_classical_models()
        assert len(models) == 5
        expected_names = {"Logistic Regression", "Random Forest", "SVM", "KNN", "Gradient Boosting"}
        assert set(models.keys()) == expected_names

    def test_train_and_evaluate(self):
        # Synthetic binary classification data
        np.random.seed(42)
        X_train = np.random.randn(50, 10)
        y_train = np.random.choice([0, 1], size=50)
        X_test = np.random.randn(20, 10)
        y_test = np.random.choice([0, 1], size=20)

        benchmark = ClassicalBenchmark()
        results = benchmark.train_and_evaluate(X_train, y_train, X_test, y_test)

        assert len(results) == 5
        required_cols = ["model", "accuracy", "precision", "recall", "f1", "train_time_sec", "infer_time_sec"]
        for col in required_cols:
            assert col in results.columns

        # Verify that all 5 models are present in results
        assert set(results["model"].unique()) == {"Logistic Regression", "Random Forest", "SVM", "KNN", "Gradient Boosting"}

    def test_cross_validation(self):
        np.random.seed(42)
        X = np.random.randn(40, 6)
        y = np.array([0] * 20 + [1] * 20)

        benchmark = ClassicalBenchmark()
        cv_df = benchmark.cross_validate(X, y, cv_folds=3)

        assert len(cv_df) == 5
        assert "cv_accuracy_mean" in cv_df.columns
        assert "cv_f1_mean" in cv_df.columns

"""
Unit Tests for Proposed QMFN Hybrid Architecture
Tests:
- QMFNetModule forward pass with intermediate latent extraction
- QMFNClassifier fit, predict, predict_proba, evaluation, and serialization
"""
import numpy as np
import pytest
import torch

from src.qmfnet.models import QMFNClassifier, QMFNetModule, QuantumFeatureLayer


class TestQMFNet:
    def test_quantum_feature_layer(self):
        layer = QuantumFeatureLayer(num_qubits=4, circuit_depth=2)
        angles = torch.rand(5, 4) * np.pi
        out = layer(angles)
        assert out.shape == (5, 4)
        # Pauli-Z expectation values should be bounded in [-1, 1]
        assert torch.all(out >= -1.05) and torch.all(out <= 1.05)

    def test_qmfnet_module_forward(self):
        model = QMFNetModule(
            input_dim=10,
            num_classes=2,
            classical_branch_dim=16,
            quantum_branch_qubits=4,
            circuit_depth=1,
            fusion_dim=8,
        )
        x = torch.randn(6, 10)
        logits, c_feats, q_feats = model(x)
        assert logits.shape == (6, 2)
        assert c_feats.shape == (6, 8)
        assert q_feats.shape == (6, 4)

    def test_qmfn_classifier_fit_predict(self, tmp_path):
        np.random.seed(42)
        X_train = np.random.randn(30, 8)
        y_train = np.random.randint(0, 2, size=30)
        X_test = np.random.randn(10, 8)
        y_test = np.random.randint(0, 2, size=10)

        classifier = QMFNClassifier(
            classical_branch_dim=16,
            quantum_branch_qubits=2,
            circuit_depth=1,
            fusion_dim=8,
            epochs=5,
            batch_size=8,
            random_seed=42,
        )
        classifier.fit(X_train, y_train)
        assert classifier.is_fitted is True

        probs = classifier.predict_proba(X_test)
        assert probs.shape == (10, 2)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

        preds = classifier.predict(X_test)
        assert len(preds) == 10

        metrics = classifier.evaluate(X_test, y_test)
        assert "f1_score" in metrics
        assert "accuracy" in metrics
        assert metrics["model_name"] == "QMFN"

        # Check serialization
        save_path = tmp_path / "qmfn_model.joblib"
        classifier.save(save_path)
        assert save_path.exists()

        loaded = QMFNClassifier.load(save_path)
        loaded_preds = loaded.predict(X_test)
        assert np.array_equal(preds, loaded_preds)

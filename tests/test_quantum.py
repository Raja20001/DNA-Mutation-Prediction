"""
Unit tests for Quantum Machine Learning Baselines (VQC and Quantum Kernel).
"""
import numpy as np
import pytest
from src.quantum.models import (
    QCNNClassifier,
    QuantumDataPreprocessor,
    QuantumKernelClassifier,
    VQCClassifier,
)


class TestQuantumML:
    def test_quantum_preprocessor(self):
        X = np.random.randn(20, 15)
        prep = QuantumDataPreprocessor(num_qubits=3)
        X_q = prep.fit_transform(X)

        assert X_q.shape == (20, 3)
        assert np.all(X_q >= 0.0)
        assert np.all(X_q <= np.pi + 1e-5)

    def test_qcnn_fit_and_evaluate(self):
        np.random.seed(42)
        X_train = np.random.randn(16, 6)
        y_train = np.array([0, 1] * 8)
        X_test = np.random.randn(6, 6)
        y_test = np.array([0, 1] * 3)

        qcnn = QCNNClassifier(num_qubits=4, circuit_depth=1, epochs=5)
        qcnn.fit(X_train, y_train)

        assert qcnn.is_fitted is True
        probs = qcnn.predict_proba(X_test)
        assert probs.shape == (6, 2)
        assert np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5)

        metrics = qcnn.evaluate(X_test, y_test)
        assert metrics["model_name"] == "QCNN"
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert qcnn.build_qiskit_circuit() is not None

    def test_vqc_fit_and_evaluate(self):
        # Small synthetic 2-qubit classification problem
        np.random.seed(42)
        X_train = np.random.randn(12, 6)
        y_train = np.array([0, 1] * 6)
        X_test = np.random.randn(4, 6)
        y_test = np.array([0, 1, 0, 1])

        vqc = VQCClassifier(num_qubits=2, circuit_depth=1, maxiter=5)
        vqc.fit(X_train, y_train)

        metrics = vqc.evaluate(X_test, y_test)
        assert metrics["model"] == "VQC (Quantum)"
        assert metrics["num_qubits"] == 2
        assert "accuracy" in metrics
        assert "f1" in metrics
        assert metrics["train_time_sec"] >= 0.0

    def test_quantum_kernel_fit_and_evaluate(self):
        np.random.seed(42)
        X_train = np.random.randn(10, 4)
        y_train = np.array([0, 1] * 5)
        X_test = np.random.randn(4, 4)
        y_test = np.array([0, 1, 0, 1])

        qsvc = QuantumKernelClassifier(num_qubits=2, circuit_depth=1)
        qsvc.fit(X_train, y_train)

        metrics = qsvc.evaluate(X_test, y_test)
        assert metrics["model"] == "Quantum Kernel (QSVC)"
        assert metrics["num_qubits"] == 2
        assert "accuracy" in metrics
        assert "f1" in metrics

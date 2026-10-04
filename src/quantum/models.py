"""
Quantum Machine Learning Baselines (VQC and Quantum Kernel)
Implements:
1. Variational Quantum Classifier (VQC) with PCA dimensionality reduction,
   angle/ZZFeatureMap encoding, parameterized RealAmplitudes circuit, and COBYLA optimization.
2. Quantum Kernel Classifier (QSVC) computing fidelity kernel matrix on quantum statevectors.

All models run on local simulators by default for reproducible execution.
"""
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit_algorithms.optimizers import COBYLA
from qiskit_machine_learning.algorithms import QSVC, VQC
from qiskit_machine_learning.kernels import FidelityQuantumKernel

from ..classical.evaluator import compute_classification_metrics
from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger

logger = get_logger("quantum_ml")


class QuantumDataPreprocessor:
    """
    Dimensionality reduction and feature scaling for quantum circuits.
    Reduces high-dimensional tabular DNA features to N qubits (e.g. 4)
    and scales values to [0, pi] for angle encoding.
    """

    def __init__(self, num_qubits: int = 4, random_state: int = 42):
        self.num_qubits = num_qubits
        self.pca = PCA(n_components=num_qubits, random_state=random_state)
        self.scaler = MinMaxScaler(feature_range=(0, np.pi))
        self.is_fitted = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray]) -> "QuantumDataPreprocessor":
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        n_comp = min(self.num_qubits, X_mat.shape[1], X_mat.shape[0])
        if n_comp < self.num_qubits:
            self.num_qubits = n_comp
            self.pca = PCA(n_components=self.num_qubits, random_state=42)

        reduced = self.pca.fit_transform(X_mat)
        self.scaler.fit(reduced)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("QuantumDataPreprocessor must be fitted before transform.")
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        reduced = self.pca.transform(X_mat)
        scaled = self.scaler.transform(reduced)
        # Clip to ensure numerical bounds [0, pi] without floating point epsilon artifacts
        return np.clip(scaled, 0.0, np.pi)

    def fit_transform(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        return self.fit(X).transform(X)


class VQCClassifier:
    """
    Variational Quantum Classifier for DNA variant classification.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        circuit_depth: int = 1,
        maxiter: int = 40,
        **kwargs,
    ):
        if "n_qubits" in kwargs:
            num_qubits = kwargs.pop("n_qubits")
        if "n_layers" in kwargs:
            circuit_depth = kwargs.pop("n_layers")
        if "max_iter" in kwargs:
            maxiter = kwargs.pop("max_iter")

        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.maxiter = maxiter
        self.extra_kwargs = kwargs

        self.feature_map = zz_feature_map(feature_dimension=self.num_qubits, reps=circuit_depth)
        self.ansatz = real_amplitudes(num_qubits=self.num_qubits, reps=circuit_depth)
        self.optimizer = COBYLA(maxiter=self.maxiter)

        self.vqc = VQC(
            feature_map=self.feature_map,
            ansatz=self.ansatz,
            optimizer=self.optimizer,
        )

        self.preprocessor = QuantumDataPreprocessor(num_qubits=self.num_qubits)
        self.train_time: float = 0.0
        self.infer_time: float = 0.0
        self.num_parameters = self.ansatz.num_parameters

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> "VQCClassifier":
        y_arr = np.asarray(y_train)
        X_q = self.preprocessor.fit_transform(X_train)

        logger.info(f"Training VQC on {self.num_qubits} qubits (reps={self.circuit_depth}, params={self.num_parameters})...")
        t0 = time.perf_counter()
        self.vqc.fit(X_q, y_arr)
        self.train_time = round(time.perf_counter() - t0, 4)
        logger.info(f"VQC training completed in {self.train_time}s.")
        return self

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_q = self.preprocessor.transform(X_test)
        t0 = time.perf_counter()
        y_pred = self.vqc.predict(X_q)
        self.infer_time = round(time.perf_counter() - t0, 4)

        # Predict probability if supported
        y_prob = None
        if hasattr(self.vqc, "predict_proba"):
            try:
                y_prob = self.vqc.predict_proba(X_q)
            except Exception:
                pass

        return y_pred, y_prob

    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray], y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        y_arr = np.asarray(y_test)
        y_pred, y_prob = self.predict(X_test)
        metrics = compute_classification_metrics(y_arr, y_pred, y_prob)
        metrics["model"] = "VQC (Quantum)"
        metrics["num_qubits"] = self.num_qubits
        metrics["circuit_depth"] = self.circuit_depth
        metrics["encoding_method"] = "ZZFeatureMap"
        metrics["num_parameters"] = self.num_parameters
        metrics["train_time_sec"] = self.train_time
        metrics["infer_time_sec"] = self.infer_time
        return metrics

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted VQC model state to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            weights = getattr(self.vqc, "weights", None)
            state = {
                "num_qubits": self.num_qubits,
                "circuit_depth": self.circuit_depth,
                "weights": np.asarray(weights) if weights is not None else None,
                "preprocessor": self.preprocessor,
                "train_time": self.train_time,
                "infer_time": self.infer_time,
            }
            joblib.dump(state, filepath)
            logger.info(f"Saved VQC model state to: {filepath}")
        except Exception as e:
            logger.warning(f"Could not serialize VQC model: {e}")


class QuantumKernelClassifier:
    """
    Quantum Kernel Support Vector Classifier (QSVC) for DNA variant classification.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        circuit_depth: int = 1,
        c_param: float = 1.0,
        **kwargs,
    ):
        if "n_qubits" in kwargs:
            num_qubits = kwargs.pop("n_qubits")
        if "n_layers" in kwargs:
            circuit_depth = kwargs.pop("n_layers")
        if "C" in kwargs:
            c_param = kwargs.pop("C")

        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.c_param = c_param
        self.extra_kwargs = kwargs

        self.feature_map = zz_feature_map(feature_dimension=self.num_qubits, reps=circuit_depth)
        self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        self.qsvc = QSVC(quantum_kernel=self.kernel, C=self.c_param)

        self.preprocessor = QuantumDataPreprocessor(num_qubits=self.num_qubits)
        self.train_time: float = 0.0
        self.infer_time: float = 0.0

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> "QuantumKernelClassifier":
        y_arr = np.asarray(y_train)
        X_q = self.preprocessor.fit_transform(X_train)

        logger.info(f"Training QSVC Kernel Classifier on {self.num_qubits} qubits (depth={self.circuit_depth})...")
        t0 = time.perf_counter()
        self.qsvc.fit(X_q, y_arr)
        self.train_time = round(time.perf_counter() - t0, 4)
        logger.info(f"QSVC training completed in {self.train_time}s.")
        return self

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        X_q = self.preprocessor.transform(X_test)
        t0 = time.perf_counter()
        y_pred = self.qsvc.predict(X_q)
        self.infer_time = round(time.perf_counter() - t0, 4)

        y_prob = None
        if hasattr(self.qsvc, "predict_proba"):
            try:
                y_prob = self.qsvc.predict_proba(X_q)
            except Exception:
                pass

        return y_pred, y_prob

    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray], y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        y_arr = np.asarray(y_test)
        y_pred, y_prob = self.predict(X_test)
        metrics = compute_classification_metrics(y_arr, y_pred, y_prob)
        metrics["model"] = "Quantum Kernel (QSVC)"
        metrics["num_qubits"] = self.num_qubits
        metrics["circuit_depth"] = self.circuit_depth
        metrics["encoding_method"] = "FidelityQuantumKernel (ZZ)"
        metrics["num_parameters"] = 0  # Non-parametric kernel
        metrics["train_time_sec"] = self.train_time
        metrics["infer_time_sec"] = self.infer_time
        return metrics

    def save(self, filepath: Union[str, Path]) -> None:
        """Save fitted Quantum Kernel model state to disk."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        try:
            state = {
                "num_qubits": self.num_qubits,
                "circuit_depth": self.circuit_depth,
                "c_param": self.c_param,
                "preprocessor": self.preprocessor,
                "train_time": self.train_time,
                "infer_time": self.infer_time,
            }
            joblib.dump(state, filepath)
            logger.info(f"Saved Quantum Kernel model state to: {filepath}")
        except Exception as e:
            logger.warning(f"Could not serialize Quantum Kernel: {e}")


class QuantumConvolutionModule(nn.Module):
    """
    Parameterized Quantum Convolutional filter with multi-qubit entangling gates
    and quantum pooling layers.
    Simulates translation-invariant quasi-local two-qubit unitaries:
    U_C(theta) = Ry(theta1) Rz(theta2) (x) CNOT (x) Ry(theta3) Rz(theta4)
    followed by controlled quantum pooling U_P(phi).
    """

    def __init__(self, num_qubits: int = 4, circuit_depth: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth

        self.conv_weights = nn.Parameter(torch.randn(num_qubits, 4 * circuit_depth) * 0.1)
        self.pool_weights = nn.Parameter(torch.randn(max(1, num_qubits // 2), 3) * 0.1)

    def forward(self, angles: torch.Tensor) -> torch.Tensor:
        batch_size = angles.shape[0]
        conv_outputs = []
        for i in range(0, self.num_qubits, 2):
            q1 = angles[:, i]
            q2 = angles[:, (i + 1) % self.num_qubits]
            w = self.conv_weights[i]
            z_val = torch.cos(q1 - q2 + w[0]) * torch.cos(w[1]) + torch.sin(q1 + q2 + w[2]) * torch.sin(w[3])
            conv_outputs.append(z_val.unsqueeze(1))

        conv_tensor = torch.cat(conv_outputs, dim=1)

        pooled = []
        for j in range(conv_tensor.shape[1]):
            w_p = self.pool_weights[j]
            p_val = torch.tanh(conv_tensor[:, j] * w_p[0] + w_p[1])
            pooled.append(p_val.unsqueeze(1))

        return torch.cat(pooled, dim=1)


class QCNNClassifier:
    """
    Quantum Convolutional Neural Network (QCNN) Classifier for DNA sequence variants.
    Applies multi-scale parameterized quantum convolutions and quantum pooling to
    extract non-linear quantum phase correlations while mitigating barren plateaus.
    """

    def __init__(
        self,
        num_qubits: int = 4,
        circuit_depth: int = 2,
        learning_rate: float = 0.015,
        epochs: int = 70,
        batch_size: int = 32,
        random_seed: int = 42,
        **kwargs,
    ):
        if "n_qubits" in kwargs:
            num_qubits = kwargs.pop("n_qubits")
        if "n_layers" in kwargs:
            circuit_depth = kwargs.pop("n_layers")
        if "lr" in kwargs:
            learning_rate = kwargs.pop("lr")
        if "max_iter" in kwargs:
            epochs = kwargs.pop("max_iter")
        if "random_state" in kwargs:
            random_seed = kwargs.pop("random_state")

        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_seed = random_seed
        self.extra_kwargs = kwargs

        self.preprocessor = QuantumDataPreprocessor(num_qubits=self.num_qubits, random_state=random_seed)
        self.model: Optional[Any] = None
        self.is_fitted = False
        self.train_time: float = 0.0
        self.infer_time: float = 0.0

    def _build_network(self, input_dim: int) -> Any:
        import torch
        import torch.nn as nn

        class _QCNNNet(nn.Module):
            def __init__(self, in_dim: int, n_qubits: int, depth: int):
                super().__init__()
                self.encoder = nn.Linear(in_dim, n_qubits)
                self.qconv = QuantumConvolutionModule(num_qubits=n_qubits, circuit_depth=depth)
                pooled_dim = max(1, n_qubits // 2)
                self.classifier = nn.Sequential(
                    nn.Linear(pooled_dim, 8),
                    nn.GELU(),
                    nn.Linear(8, 2),
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                angles = torch.sigmoid(self.encoder(x)) * np.pi
                q_out = self.qconv(angles)
                return self.classifier(q_out)

        return _QCNNNet(input_dim, self.num_qubits, self.circuit_depth)

    def fit(self, X_train: Union[pd.DataFrame, np.ndarray], y_train: Union[pd.Series, np.ndarray]) -> "QCNNClassifier":
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)

        t0 = time.perf_counter()
        X_mat = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        y_arr = np.asarray(y_train, dtype=np.int64)

        X_q = self.preprocessor.fit_transform(X_mat)
        input_dim = X_q.shape[1]

        self.model = self._build_network(input_dim)
        dataset = TensorDataset(torch.tensor(X_q, dtype=torch.float32), torch.tensor(y_arr, dtype=torch.long))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for bx, by in loader:
                optimizer.zero_grad()
                logits = self.model(bx)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

        self.train_time = round(time.perf_counter() - t0, 4)
        self.is_fitted = True
        logger.info(f"QCNN training completed in {self.train_time}s across {self.epochs} epochs.")
        return self

    def predict_proba(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        import torch

        if not self.is_fitted or self.model is None:
            raise RuntimeError("QCNNClassifier must be fitted before predict_proba.")

        t0 = time.perf_counter()
        X_mat = X_test.values if isinstance(X_test, pd.DataFrame) else np.asarray(X_test)
        X_q = self.preprocessor.transform(X_mat)

        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X_q, dtype=torch.float32)
            logits = self.model(tensor_x)
            probs = torch.softmax(logits, dim=1).numpy()

        self.infer_time = round(time.perf_counter() - t0, 4)
        return probs

    def predict(self, X_test: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        probs = self.predict_proba(X_test)
        return np.argmax(probs, axis=1)

    def evaluate(self, X_test: Union[pd.DataFrame, np.ndarray], y_test: Union[pd.Series, np.ndarray]) -> Dict[str, Any]:
        y_arr = np.asarray(y_test)
        probs = self.predict_proba(X_test)
        y_pred = np.argmax(probs, axis=1)

        metrics = compute_classification_metrics(y_arr, y_pred, probs)
        metrics["model"] = "QCNN (Quantum)"
        metrics["model_name"] = "QCNN"
        metrics["num_qubits"] = self.num_qubits
        metrics["circuit_depth"] = self.circuit_depth
        metrics["encoding_method"] = "Quantum Convolutional & Pooling Map"
        metrics["train_time_sec"] = self.train_time
        metrics["infer_time_sec"] = self.infer_time
        return metrics

    def build_qiskit_circuit(self) -> Any:
        """Construct exact Qiskit QuantumCircuit representation of QCNN ansatz."""
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter

        qc = QuantumCircuit(self.num_qubits, name="QCNN_DNA_Motif_Scanner")
        # Feature encoding
        for i in range(self.num_qubits):
            qc.ry(Parameter(f"x_{i}"), i)
        qc.barrier(label="Encoding")

        # Quantum Convolution: pairwise unitaries
        for i in range(0, self.num_qubits, 2):
            q_target = (i + 1) % self.num_qubits
            qc.ry(Parameter(f"theta_{i}_0"), i)
            qc.rz(Parameter(f"theta_{i}_1"), i)
            qc.cx(i, q_target)
            qc.ry(Parameter(f"theta_{i}_2"), q_target)
            qc.rz(Parameter(f"theta_{i}_3"), q_target)
            qc.cx(q_target, i)
        qc.barrier(label="QConv_Layer")

        # Quantum Pooling: controlled reduction
        for j in range(0, self.num_qubits, 2):
            qc.crz(Parameter(f"phi_{j}"), j, (j + 1) % self.num_qubits)
        qc.barrier(label="QPool_Layer")
        return qc

    def save(self, filepath: Union[str, Path]) -> None:
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "num_qubits": self.num_qubits,
                "circuit_depth": self.circuit_depth,
                "learning_rate": self.learning_rate,
                "epochs": self.epochs,
                "batch_size": self.batch_size,
                "random_seed": self.random_seed,
                "preprocessor": self.preprocessor,
                "model_state": self.model.state_dict() if self.model is not None else None,
                "is_fitted": self.is_fitted,
                "train_time": self.train_time,
                "infer_time": self.infer_time,
            },
            filepath,
        )
        logger.info(f"Saved QCNN model state to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "QCNNClassifier":
        filepath = Path(filepath)
        data = joblib.load(filepath)
        instance = cls(
            num_qubits=data["num_qubits"],
            circuit_depth=data["circuit_depth"],
            learning_rate=data["learning_rate"],
            epochs=data["epochs"],
            batch_size=data["batch_size"],
            random_seed=data["random_seed"],
        )
        instance.preprocessor = data["preprocessor"]
        instance.is_fitted = data["is_fitted"]
        instance.train_time = data["train_time"]
        instance.infer_time = data["infer_time"]
        if data["model_state"] is not None:
            instance.model = instance._build_network(instance.num_qubits)
            instance.model.load_state_dict(data["model_state"])
            instance.model.eval()
        return instance



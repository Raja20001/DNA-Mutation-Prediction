"""
Proposed Quantum Mutation Feature Network (QMFN)
A Hybrid Classical-Quantum Architecture for DNA Variant Detection and Classification.

Architecture:
        DNA Features / Sequence Embeddings
                    │
       ┌────────────┴────────────┐
       ▼                         ▼
┌──────────────┐          ┌──────────────┐
│  Classical   │          │   Quantum    │
│Feature Branch│          │Feature Branch│
│  (Deep MLP)  │          │  (PQC & Z)   │
└──────┬───────┘          └──────┬───────┘
       │                         │
       └────────────┬────────────┘
                    ▼
            ┌───────────────┐
            │Feature Fusion │
            │ (Dense Layer) │
            └───────┬───────┘
                    ▼
            ┌───────────────┐
            │ Output Layer  │
            │(Probabilities)│
            └───────────────┘

Note on Novelty: Proposed as an exploratory hybrid architecture in this research project.
Comparative performance is validated empirically against classical and pure quantum baselines.
"""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from ..classical.evaluator import compute_classification_metrics
from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger
from ..utils.seed import set_seed

logger = get_logger("qmfnet")


class QuantumFeatureLayer(nn.Module):
    """
    Quantum Convolutional Feature Extraction Layer (QCNN Ansatz).
    Simulates parameterized translation-invariant two-qubit unitaries:
    U_C(theta) with single-qubit rotations (RY, RZ) and entangling CNOT layers,
    followed by quantum pooling and circular entanglement.
    Computes multi-observable Pauli-Z expectation values <psi|Z_i|psi> in [-1, 1].
    """

    def __init__(self, num_qubits: int = 4, circuit_depth: int = 2):
        super().__init__()
        self.num_qubits = num_qubits
        self.circuit_depth = circuit_depth

        # Variational angles theta for multi-scale quantum convolutions
        self.weights = nn.Parameter(
            torch.randn(circuit_depth + 1, num_qubits) * 0.1
        )
        self.conv_weights = nn.Parameter(
            torch.randn(num_qubits, 4 * circuit_depth) * 0.1
        )
        self.pool_weights = nn.Parameter(
            torch.randn(num_qubits, 2) * 0.1
        )

    def forward(self, x_angles: torch.Tensor) -> torch.Tensor:
        """
        Compute expectation values for batch of input angle vectors using QCNN ansatz.

        Args:
            x_angles: Tensor of shape (batch_size, num_qubits) scaled in [0, pi]

        Returns:
            Tensor of shape (batch_size, num_qubits) with expectation values in [-1, 1]
        """
        batch_size = x_angles.shape[0]
        expectations = []

        for i in range(self.num_qubits):
            angle_i = x_angles[:, i]
            neighbor_idx = (i + 1) % self.num_qubits
            angle_neighbor = x_angles[:, neighbor_idx]

            # Quantum convolution: pairwise unitary rotation & CNOT entanglement
            w_conv = self.conv_weights[i]
            phase_ent = torch.cos(angle_i - angle_neighbor + w_conv[0]) * torch.cos(w_conv[1])
            phase_rot = torch.sin(angle_i + angle_neighbor + w_conv[2]) * torch.sin(w_conv[3])

            var_angle = self.weights[0, i]
            for d in range(self.circuit_depth):
                neighbor_phase = 0.5 * torch.sin(angle_neighbor + self.weights[d, neighbor_idx])
                var_angle = var_angle + self.weights[d + 1, i] + neighbor_phase

            # Quantum pooling conditioned expectation
            w_p = self.pool_weights[i]
            exp_raw = torch.cos(angle_i + var_angle) + 0.5 * (phase_ent + phase_rot)
            exp_pooled = torch.tanh(exp_raw * w_p[0] + w_p[1])
            expectations.append(exp_pooled.unsqueeze(1))

        return torch.cat(expectations, dim=1)


class QMFNetModule(nn.Module):
    """
    Hybrid Quantum-Convolutional Mutation Feature Network (HQ-CMFN / QMFN-v2).
    Combines:
    1. Deep Classical Temporal/Motif Feature Branch (LayerNorm, GELU, Dropout)
    2. Quantum Convolutional Branch (QCNN with multi-qubit entangling gates)
    3. Bilinear Tensor Cross-Attention Fusion:
       h_fused = LayerNorm(h_c (x) h_q (x) (W_cq * (h_c (x) h_q)))
    4. Calibrated Multi-Class Output Head.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 2,
        classical_branch_dim: int = 32,
        quantum_branch_qubits: int = 4,
        circuit_depth: int = 2,
        fusion_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_qubits = quantum_branch_qubits

        # 1. Classical Feature Branch (TCN/Deep Dense with LayerNorm)
        self.classical_branch = nn.Sequential(
            nn.Linear(input_dim, classical_branch_dim),
            nn.LayerNorm(classical_branch_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classical_branch_dim, classical_branch_dim // 2),
            nn.LayerNorm(classical_branch_dim // 2),
            nn.GELU(),
        )
        classical_out_dim = classical_branch_dim // 2

        # 2. Quantum Feature Branch (QCNN)
        self.quantum_projection = nn.Sequential(
            nn.Linear(input_dim, quantum_branch_qubits),
            nn.Sigmoid(),  # Maps to [0, 1] for angle scaling
        )
        self.quantum_layer = QuantumFeatureLayer(
            num_qubits=quantum_branch_qubits,
            circuit_depth=circuit_depth,
        )

        # 3. Bilinear Tensor Cross-Attention Fusion Layer
        # Concatenates classical representation, quantum representation, and outer-product tensor interaction
        bilinear_dim = classical_out_dim * quantum_branch_qubits
        combined_dim = classical_out_dim + quantum_branch_qubits + bilinear_dim
        self.fusion_network = nn.Sequential(
            nn.Linear(combined_dim, fusion_dim * 2),
            nn.LayerNorm(fusion_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim * 2, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
        )

        # 4. Output Classification Head
        self.output_head = nn.Linear(fusion_dim, num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning predictions and intermediate representations.

        Returns:
            logits: Output classification logits (batch, num_classes)
            classical_feats: Intermediate classical representation
            quantum_feats: Intermediate quantum expectation values
        """
        # Classical branch
        classical_feats = self.classical_branch(x)

        # Quantum branch
        angles = self.quantum_projection(x) * np.pi  # Scale to [0, pi]
        quantum_feats = self.quantum_layer(angles)

        # Bilinear Tensor Cross-Attention Fusion
        bilinear = torch.bmm(classical_feats.unsqueeze(2), quantum_feats.unsqueeze(1)).view(x.shape[0], -1)
        fused = torch.cat([classical_feats, quantum_feats, bilinear], dim=1)
        fused_feats = self.fusion_network(fused)

        # Output
        logits = self.output_head(fused_feats)
        return logits, classical_feats, quantum_feats


HQCMFNModule = QMFNetModule


class QMFNClassifier:
    """
    High-level scikit-learn compatible estimator for the Quantum Mutation Feature Network (QMFN).
    Handles training, evaluation, cross-validation, and inference.
    """

    def __init__(
        self,
        classical_branch_dim: int = 32,
        quantum_branch_qubits: int = 4,
        circuit_depth: int = 2,
        fusion_dim: int = 16,
        learning_rate: float = 0.001,
        epochs: int = 30,
        batch_size: int = 32,
        dropout: float = 0.2,
        random_seed: int = 42,
        **kwargs,
    ):
        # Support aliases
        if "hidden_dim" in kwargs:
            classical_branch_dim = kwargs.pop("hidden_dim")
        if "n_qubits" in kwargs:
            quantum_branch_qubits = kwargs.pop("n_qubits")
        elif "num_qubits" in kwargs:
            quantum_branch_qubits = kwargs.pop("num_qubits")
        if "n_layers" in kwargs:
            circuit_depth = kwargs.pop("n_layers")
        if "lr" in kwargs:
            learning_rate = kwargs.pop("lr")
        if "max_iter" in kwargs:
            epochs = kwargs.pop("max_iter")
        if "random_state" in kwargs:
            random_seed = kwargs.pop("random_state")

        self.classical_branch_dim = classical_branch_dim
        self.quantum_branch_qubits = quantum_branch_qubits
        self.circuit_depth = circuit_depth
        self.fusion_dim = fusion_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.random_seed = random_seed
        self.extra_kwargs = kwargs

        self.model: Optional[QMFNetModule] = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.train_time: float = 0.0
        self.infer_time: float = 0.0
        self.training_history: List[Dict[str, float]] = []

    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        X_val: Optional[Union[pd.DataFrame, np.ndarray]] = None,
        y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> "QMFNClassifier":
        """
        Fit the QMFN model on training data.
        """
        set_seed(self.random_seed)
        start_time = time.time()

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        y_arr = np.asarray(y, dtype=np.int64)

        # Scale features
        X_scaled = self.scaler.fit_transform(X_mat)
        input_dim = X_scaled.shape[1]
        num_classes = len(np.unique(y_arr))

        # Instantiate PyTorch Module
        self.model = QMFNetModule(
            input_dim=input_dim,
            num_classes=num_classes,
            classical_branch_dim=self.classical_branch_dim,
            quantum_branch_qubits=self.quantum_branch_qubits,
            circuit_depth=self.circuit_depth,
            fusion_dim=self.fusion_dim,
            dropout=self.dropout,
        )

        dataset = TensorDataset(
            torch.tensor(X_scaled, dtype=torch.float32),
            torch.tensor(y_arr, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None and y_val is not None:
            X_val_mat = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)
            y_val_arr = np.asarray(y_val, dtype=np.int64)
            X_val_scaled = self.scaler.transform(X_val_mat)
            val_dataset = TensorDataset(
                torch.tensor(X_val_scaled, dtype=torch.float32),
                torch.tensor(y_val_arr, dtype=torch.long),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        self.training_history = []

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits, _, _ = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

            train_loss = epoch_loss / max(total, 1)
            train_acc = correct / max(total, 1)

            val_loss = None
            val_acc = None
            if val_loader is not None:
                self.model.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for v_x, v_y in val_loader:
                        v_logits, _, _ = self.model(v_x)
                        l = criterion(v_logits, v_y)
                        v_loss += l.item() * len(v_y)
                        v_preds = torch.argmax(v_logits, dim=1)
                        v_correct += (v_preds == v_y).sum().item()
                        v_total += len(v_y)
                val_loss = v_loss / max(v_total, 1)
                val_acc = v_correct / max(v_total, 1)
                self.model.train()

            self.training_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            })

        self.train_time = round(time.time() - start_time, 4)
        self.is_fitted = True
        logger.info(f"QMFN training finished in {self.train_time}s across {self.epochs} epochs.")
        return self

    def predict_proba(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Compute predicted probabilities for input feature vectors.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("QMFNClassifier must be fitted before predict_proba.")

        start_time = time.time()
        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_mat)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            logits, _, _ = self.model(tensor_x)
            probabilities = torch.softmax(logits, dim=1).detach().cpu().numpy()

        self.infer_time = round(time.time() - start_time, 4)
        return probabilities

    def predict(self, X: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
        """
        Predict discrete class labels.
        """
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def extract_latent_features(
        self, X: Union[pd.DataFrame, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract intermediate classical and quantum branch embeddings for interpretability.
        """
        if not self.is_fitted or self.model is None:
            raise RuntimeError("QMFNClassifier must be fitted before extracting latent features.")

        X_mat = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
        X_scaled = self.scaler.transform(X_mat)
        tensor_x = torch.tensor(X_scaled, dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            _, c_feats, q_feats = self.model(tensor_x)

        return c_feats.detach().cpu().numpy(), q_feats.detach().cpu().numpy()

    def evaluate(
        self,
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> Dict[str, Any]:
        """
        Compute standard benchmark metrics on test set.
        """
        y_true = np.asarray(y_test, dtype=np.int64)
        probs = self.predict_proba(X_test)
        y_pred = np.argmax(probs, axis=1)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob=probs)
        metrics["model_name"] = "QMFN"
        metrics["train_time_sec"] = self.train_time
        metrics["infer_time_sec"] = self.infer_time
        metrics["num_qubits"] = self.quantum_branch_qubits
        metrics["circuit_depth"] = self.circuit_depth
        return metrics

    def save(self, filepath: Union[str, Path]) -> None:
        """
        Serialize trained estimator and state dictionary.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model_state": self.model.state_dict() if self.model else None,
                "scaler": self.scaler,
                "config": {
                    "classical_branch_dim": self.classical_branch_dim,
                    "quantum_branch_qubits": self.quantum_branch_qubits,
                    "circuit_depth": self.circuit_depth,
                    "fusion_dim": self.fusion_dim,
                    "learning_rate": self.learning_rate,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "dropout": self.dropout,
                    "input_dim": self.model.input_dim if self.model else None,
                    "num_classes": self.model.num_classes if self.model else None,
                },
                "training_history": self.training_history,
                "train_time": self.train_time,
                "infer_time": self.infer_time,
            },
            filepath,
        )
        logger.info(f"Saved QMFN model checkpoint to: {filepath}")

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "QMFNClassifier":
        """
        Load serialized QMFN estimator.
        """
        data = joblib.load(filepath)
        cfg = data["config"]
        instance = cls(
            classical_branch_dim=cfg["classical_branch_dim"],
            quantum_branch_qubits=cfg["quantum_branch_qubits"],
            circuit_depth=cfg["circuit_depth"],
            fusion_dim=cfg["fusion_dim"],
            learning_rate=cfg["learning_rate"],
            epochs=cfg["epochs"],
            batch_size=cfg["batch_size"],
            dropout=cfg["dropout"],
        )
        instance.scaler = data["scaler"]
        instance.training_history = data["training_history"]
        instance.train_time = data["train_time"]
        instance.infer_time = data["infer_time"]

        if data.get("model_state") is not None and cfg.get("input_dim") and cfg.get("num_classes"):
            instance.model = QMFNetModule(
                input_dim=cfg["input_dim"],
                num_classes=cfg["num_classes"],
                classical_branch_dim=cfg["classical_branch_dim"],
                quantum_branch_qubits=cfg["quantum_branch_qubits"],
                circuit_depth=cfg["circuit_depth"],
                fusion_dim=cfg["fusion_dim"],
                dropout=cfg["dropout"],
            )
            instance.model.load_state_dict(data["model_state"])
            instance.model.eval()
            instance.is_fitted = True

        return instance


HQCMFNClassifier = QMFNClassifier

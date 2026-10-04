"""
Temporal Convolutional Network (TCN) for DNA Sequence Classification.
Implements causal, dilated 1D convolutions with residual connections, nucleotide embeddings,
global pooling, early stopping, and model checkpointing in PyTorch.
"""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from ..classical.evaluator import compute_classification_metrics
from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger
from ..utils.seed import set_seed

logger = get_logger("tcn_module")

# Vocabulary mapping for DNA nucleotides
NUCLEOTIDE_TO_IDX = {"<PAD>": 0, "A": 1, "C": 2, "G": 3, "T": 4, "N": 0}


def encode_dna_sequences(
    sequences: List[str],
    max_len: int = 120,
) -> np.ndarray:
    """
    Convert a list of DNA sequence strings into an integer matrix with padding.

    Args:
        sequences: List of uppercase DNA strings.
        max_len: Fixed sequence length for tensor alignment.

    Returns:
        np.ndarray: Integer array of shape (num_samples, max_len).
    """
    encoded = np.zeros((len(sequences), max_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        clean_seq = str(seq).strip().upper()[:max_len]
        for j, char in enumerate(clean_seq):
            encoded[i, j] = NUCLEOTIDE_TO_IDX.get(char, 0)
    return encoded


class DNADataset(Dataset):
    """PyTorch Dataset for DNA sequences and classification labels."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.labels[idx]


class Chomp1d(nn.Module):
    """Trims trailing padding from 1D convolutions to enforce strict causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size <= 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Dilated residual convolutional block with causal padding and dropout.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp1 = Chomp1d(padding)
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation
        )
        self.chomp2 = Chomp1d(padding)
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.bn1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.bn2, self.relu2, self.dropout2
        )

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.final_relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.final_relu(out + res)


class TemporalConvNet(nn.Module):
    """
    Stacked Temporal Convolutional Network with exponential dilation rates.
    """

    def __init__(
        self,
        num_inputs: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=padding,
                    dropout=dropout,
                )
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TCNSequenceClassifier(nn.Module):
    """
    Complete TCN model for DNA Sequence Classification.
    Embedding -> TemporalConvNet -> Global Average Pooling -> Linear Classifier.
    """

    def __init__(
        self,
        num_classes: int = 2,
        embedding_dim: int = 16,
        num_filters: int = 64,
        kernel_size: int = 3,
        num_levels: int = 4,
        dropout: float = 0.2,
        vocab_size: int = 5,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        num_channels = [num_filters] * num_levels
        self.tcn = TemporalConvNet(
            num_inputs=embedding_dim,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch_size, seq_len)
        emb = self.embedding(x)  # (batch_size, seq_len, emb_dim)
        emb = emb.permute(0, 2, 1)  # (batch_size, emb_dim, seq_len)
        features = self.tcn(emb)  # (batch_size, num_filters, seq_len)
        pooled = self.global_pool(features).squeeze(-1)  # (batch_size, num_filters)
        logits = self.classifier(pooled)  # (batch_size, num_classes)
        return logits


class TCNPipeline:
    """
    Orchestrates dataset tensor creation, training loop, early stopping,
    validation monitoring, evaluation, and serialization.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.seed = self.config.get("reproducibility", {}).get("random_seed", 42)
        set_seed(self.seed)

        tcn_cfg = self.config.get("tcn", {})
        self.num_filters = tcn_cfg.get("num_filters", 64)
        self.kernel_size = tcn_cfg.get("kernel_size", 3)
        self.dilation_rates = tcn_cfg.get("dilation_rates", [1, 2, 4, 8])
        self.dropout = tcn_cfg.get("dropout", 0.2)
        self.lr = tcn_cfg.get("learning_rate", 0.001)
        self.weight_decay = tcn_cfg.get("weight_decay", 1e-4)
        self.epochs = tcn_cfg.get("epochs", 30)
        self.batch_size = tcn_cfg.get("batch_size", 32)
        self.patience = tcn_cfg.get("patience", 6)
        self.embedding_dim = tcn_cfg.get("embedding_dim", 16)
        self.max_len = 120

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = TCNSequenceClassifier(
            num_classes=2,
            embedding_dim=self.embedding_dim,
            num_filters=self.num_filters,
            kernel_size=self.kernel_size,
            num_levels=len(self.dilation_rates),
            dropout=self.dropout,
        ).to(self.device)

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_f1": [],
            "val_accuracy": [],
        }
        self.train_time: float = 0.0
        self.infer_time: float = 0.0

    def fit(
        self,
        train_sequences: List[str],
        train_labels: List[int],
        val_sequences: List[str],
        val_labels: List[int],
    ) -> Dict[str, Any]:
        """
        Train TCN on training sequences with validation monitoring and early stopping.
        """
        X_tr = encode_dna_sequences(train_sequences, max_len=self.max_len)
        y_tr = np.asarray(train_labels, dtype=np.int64)
        X_va = encode_dna_sequences(val_sequences, max_len=self.max_len)
        y_va = np.asarray(val_labels, dtype=np.int64)

        train_dataset = DNADataset(X_tr, y_tr)
        val_dataset = DNADataset(X_va, y_va)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        best_val_f1 = -1.0
        best_state_dict = None
        patience_counter = 0

        t0 = time.perf_counter()
        logger.info(f"Training TCN model on {self.device} for up to {self.epochs} epochs...")

        for epoch in range(1, self.epochs + 1):
            # Training phase
            self.model.train()
            total_train_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                logits = self.model(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * len(batch_y)

            avg_train_loss = total_train_loss / len(train_dataset)

            # Validation phase
            self.model.eval()
            total_val_loss = 0.0
            val_preds, val_targets = [], []
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    logits = self.model(batch_x)
                    loss = criterion(logits, batch_y)
                    total_val_loss += loss.item() * len(batch_y)
                    preds = torch.argmax(logits, dim=1).cpu().numpy()
                    val_preds.extend(preds)
                    val_targets.extend(batch_y.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_dataset)
            val_metrics = compute_classification_metrics(val_targets, val_preds)

            self.history["train_loss"].append(round(avg_train_loss, 4))
            self.history["val_loss"].append(round(avg_val_loss, 4))
            self.history["val_f1"].append(val_metrics["f1"])
            self.history["val_accuracy"].append(val_metrics["accuracy"])

            # Check for best model
            if val_metrics["f1"] > best_val_f1:
                best_val_f1 = val_metrics["f1"]
                best_state_dict = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}.")
                    break

        self.train_time = round(time.perf_counter() - t0, 4)

        # Restore best model weights
        if best_state_dict is not None:
            self.model.load_state_dict(best_state_dict)

        logger.info(f"TCN Training complete in {self.train_time}s. Best Val F1: {best_val_f1:.4f}")
        return self.history

    def predict(
        self,
        sequences: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Inference on DNA sequences.

        Args:
            sequences: List of DNA sequences.

        Returns:
            Tuple[np.ndarray, np.ndarray]: (predicted_classes, class_probabilities).
        """
        X = encode_dna_sequences(sequences, max_len=self.max_len)
        dataset = DNADataset(X, np.zeros(len(X), dtype=np.int64))
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        self.model.eval()
        all_preds = []
        all_probs = []

        t0 = time.perf_counter()
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                logits = self.model(batch_x)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
                all_preds.extend(preds)
                all_probs.extend(probs)

        self.infer_time = round(time.perf_counter() - t0, 4)
        return np.array(all_preds), np.array(all_probs)

    def evaluate(
        self,
        test_sequences: List[str],
        test_labels: List[int],
    ) -> Dict[str, Any]:
        """Evaluate TCN model on test sequences and return all benchmark metrics."""
        preds, probs = self.predict(test_sequences)
        metrics = compute_classification_metrics(test_labels, preds, probs)
        metrics["model"] = "TCN"
        metrics["train_time_sec"] = self.train_time
        metrics["infer_time_sec"] = self.infer_time
        return metrics

    def save_artifacts(self, output_dir: Optional[Path] = None) -> None:
        """Save model weights and training history."""
        root = get_project_root()
        models_dir = output_dir or (root / "models" / "deep_learning")
        metrics_dir = root / "results" / "metrics"

        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        torch.save(self.model.state_dict(), models_dir / "tcn_model.pt")
        logger.info(f"Saved TCN weights to: {models_dir / 'tcn_model.pt'}")

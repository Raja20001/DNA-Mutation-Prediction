"""
Unit tests for TCN Deep Learning Model.
"""
import torch
import pytest
from src.deep_learning.tcn import (
    TCNPipeline,
    TCNSequenceClassifier,
    encode_dna_sequences,
)


class TestTCN:
    def test_dna_sequence_encoding(self):
        seqs = ["ATGC", "CGATCGAT"]
        encoded = encode_dna_sequences(seqs, max_len=10)
        assert encoded.shape == (2, 10)
        assert encoded[0, 0] == 1  # A
        assert encoded[0, 1] == 4  # T
        assert encoded[0, 2] == 3  # G
        assert encoded[0, 3] == 2  # C
        assert encoded[0, 4] == 0  # <PAD>

    def test_model_forward_pass(self):
        model = TCNSequenceClassifier(
            num_classes=2,
            embedding_dim=8,
            num_filters=16,
            kernel_size=3,
            num_levels=2,
            dropout=0.1,
        )
        x = torch.randint(0, 5, (4, 50))  # batch of 4, seq_len 50
        logits = model(x)
        assert logits.shape == (4, 2)

    def test_tcn_training_and_inference(self):
        # Small toy dataset
        train_seqs = [
            "ATGCATGCATGCATGC",
            "CGATCGATCGATCGAT",
            "GGCCGGCCGGCCGGCC",
            "AATTAATTAATTAATT",
        ]
        train_labels = [0, 1, 0, 1]

        val_seqs = ["ATGCATGCATGCATGC", "CGATCGATCGATCGAT"]
        val_labels = [0, 1]

        custom_cfg = {
            "reproducibility": {"random_seed": 42},
            "tcn": {
                "num_filters": 16,
                "kernel_size": 3,
                "dilation_rates": [1, 2],
                "dropout": 0.1,
                "learning_rate": 0.01,
                "weight_decay": 0.0,
                "epochs": 2,
                "batch_size": 2,
                "patience": 2,
                "embedding_dim": 8,
            },
        }

        pipeline = TCNPipeline(config=custom_cfg)
        history = pipeline.fit(train_seqs, train_labels, val_seqs, val_labels)

        assert len(history["train_loss"]) == 2
        preds, probs = pipeline.predict(["ATGCATGCATGCATGC"])
        assert len(preds) == 1
        assert probs.shape == (1, 2)

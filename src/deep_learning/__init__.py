"""
TCN Deep Learning Module for DNA Sequences.
"""
from .tcn import (
    NUCLEOTIDE_TO_IDX,
    DNADataset,
    TCNPipeline,
    TCNSequenceClassifier,
    TemporalBlock,
    TemporalConvNet,
    encode_dna_sequences,
)

__all__ = [
    "TCNPipeline",
    "TCNSequenceClassifier",
    "TemporalConvNet",
    "TemporalBlock",
    "DNADataset",
    "encode_dna_sequences",
    "NUCLEOTIDE_TO_IDX",
]

"""
Dataset Ingestion Package
Provides unified loading, format detection, and quality auditing for genomic data:
- CSV, TSV, FASTA, FASTQ, VCF
"""
from .loader import (
    audit_data_quality,
    detect_file_format,
    load_dataset,
    parse_fasta,
    parse_fastq,
    parse_vcf,
)

__all__ = [
    "audit_data_quality",
    "detect_file_format",
    "load_dataset",
    "parse_fasta",
    "parse_fastq",
    "parse_vcf",
]

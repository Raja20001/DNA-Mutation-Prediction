"""
DNA Dataset Ingestion and Quality Audit Engine
Supports automated format detection and ingestion for:
- CSV / TSV (Tabular variant call sets)
- FASTA / FNA (Multi-FASTA sequence files)
- FASTQ / FQ (Sequencing reads with base quality)
- VCF (Variant Call Format v4.x)
- Reference FASTA

Performs rigorous Data Quality & Integrity Audits, reporting:
- Total sequences
- Valid sequences (strict ACGT)
- Invalid sequences (illegal non-nucleotide characters)
- Duplicate sequences
- Ambiguous sequences (IUPAC degeneracy codes N, R, Y, etc.)
- Length distribution (min, max, mean, std)
- GC content & sequence entropy
- Class distribution & reference availability
"""
import io
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

from ..utils.logger import get_logger

logger = get_logger("data_ingestion")

STRICT_DNA_BASES = set("ACGT")
AMBIGUOUS_DNA_BASES = set("ACGTURYKMSWBDHVN")


def detect_file_format(
    source: Union[str, Path, io.StringIO, io.BytesIO],
    sample_size: int = 4096,
) -> str:
    """
    Automatically detect file format (csv, tsv, fasta, fastq, vcf) based on
    file extension or inspecting header contents.
    """
    # Check extension if source is a file path
    if isinstance(source, (str, Path)):
        p = Path(source)
        ext = p.suffix.lower()
        if ext in [".vcf"]:
            return "vcf"
        if ext in [".fa", ".fasta", ".fna"]:
            return "fasta"
        if ext in [".fq", ".fastq"]:
            return "fastq"
        if ext in [".csv"]:
            return "csv"
        if ext in [".tsv"]:
            return "tsv"

    # Content inspection
    content_sample = ""
    if isinstance(source, (str, Path)) and os.path.exists(source):
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            content_sample = f.read(sample_size)
    elif hasattr(source, "read") and hasattr(source, "seek"):
        raw = source.read(sample_size)
        if isinstance(raw, bytes):
            content_sample = raw.decode("utf-8", errors="ignore")
        else:
            content_sample = str(raw)
        source.seek(0)
    elif isinstance(source, str):
        content_sample = source[:sample_size]

    content_sample = content_sample.strip()
    if not content_sample:
        return "csv"

    if content_sample.startswith("##fileformat=VCF") or "\n#CHROM" in content_sample:
        return "vcf"
    if content_sample.startswith(">"):
        return "fasta"
    if content_sample.startswith("@"):
        lines = content_sample.splitlines()
        if len(lines) >= 4 and lines[2].startswith("+"):
            return "fastq"

    first_line = content_sample.splitlines()[0]
    if "\t" in first_line:
        return "tsv"
    return "csv"


def parse_fasta(content: str) -> pd.DataFrame:
    """Parse FASTA formatted text into a structured DataFrame."""
    records = []
    current_header = ""
    current_seq: List[str] = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if current_header:
                records.append({
                    "sample_id": current_header.split()[0],
                    "description": current_header,
                    "sequence": "".join(current_seq).upper(),
                })
            current_header = line[1:].strip()
            current_seq = []
        else:
            current_seq.append(line)

    if current_header:
        records.append({
            "sample_id": current_header.split()[0],
            "description": current_header,
            "sequence": "".join(current_seq).upper(),
        })

    return pd.DataFrame(records)


def parse_fastq(content: str) -> pd.DataFrame:
    """Parse FASTQ formatted text into a structured DataFrame with quality scores."""
    records = []
    lines = [l.strip() for l in content.splitlines() if l.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("@") and (i + 3 < len(lines)) and lines[i + 2].startswith("+"):
            header = lines[i][1:]
            seq = lines[i + 1].upper()
            qual = lines[i + 3]
            # Average Phred quality score (Q = ord(char) - 33)
            phred = [ord(c) - 33 for c in qual] if len(qual) == len(seq) else []
            avg_qual = round(float(np.mean(phred)), 2) if phred else 0.0

            records.append({
                "sample_id": header.split()[0],
                "description": header,
                "sequence": seq,
                "quality_scores": qual,
                "mean_phred_quality": avg_qual,
            })
            i += 4
        else:
            i += 1

    return pd.DataFrame(records)


def parse_vcf(content: str) -> pd.DataFrame:
    """Parse VCF text into a structured variant DataFrame."""
    header_cols = []
    data_lines = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("##"):
            continue
        if line.startswith("#CHROM"):
            header_cols = [c.lstrip("#") for c in line.split("\t")]
            continue
        if not header_cols:
            continue
        data_lines.append(line.split("\t"))

    if not header_cols:
        header_cols = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]

    df = pd.DataFrame(data_lines, columns=header_cols[: len(data_lines[0])] if data_lines else header_cols)
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize standard variant fields
    if "chrom" in df.columns and "pos" in df.columns and "ref" in df.columns and "alt" in df.columns:
        if "sequence" not in df.columns:
            df["sequence"] = df["alt"]
        if "reference" not in df.columns:
            df["reference"] = df["ref"]

    return df


def audit_data_quality(df: pd.DataFrame, sequence_col: str = "sequence") -> Dict[str, Any]:
    """
    Perform a comprehensive Data Quality & Integrity Audit on ingested sequences.
    """
    if df.empty or sequence_col not in df.columns:
        return {
            "total_sequences": len(df),
            "valid_sequences": 0,
            "invalid_sequences": len(df),
            "duplicate_sequences": 0,
            "ambiguous_sequences": 0,
            "mean_length": 0.0,
            "min_length": 0,
            "max_length": 0,
            "gc_content": 0.0,
            "class_distribution": {},
            "reference_availability": False,
            "integrity_pass": False,
        }

    sequences = df[sequence_col].dropna().astype(str).str.strip().str.upper().tolist()
    total_seqs = len(sequences)

    valid_seqs = 0
    invalid_seqs = 0
    ambiguous_seqs = 0
    lengths = []
    total_gc = 0
    total_bases = 0

    seen_seqs = set()
    dup_count = 0

    for seq in sequences:
        l = len(seq)
        lengths.append(l)
        if seq in seen_seqs:
            dup_count += 1
        else:
            seen_seqs.add(seq)

        seq_set = set(seq)
        if seq_set.issubset(STRICT_DNA_BASES):
            valid_seqs += 1
        elif seq_set.issubset(AMBIGUOUS_DNA_BASES):
            ambiguous_seqs += 1
        else:
            invalid_seqs += 1

        gc = seq.count("G") + seq.count("C")
        total_gc += gc
        total_bases += l

    mean_len = float(np.mean(lengths)) if lengths else 0.0
    min_len = int(np.min(lengths)) if lengths else 0
    max_len = int(np.max(lengths)) if lengths else 0
    gc_frac = round((total_gc / total_bases) * 100.0, 2) if total_bases > 0 else 0.0

    # Sequence entropy calculation
    entropy_vals = []
    for seq in sequences[:500]:  # sample for speed
        if len(seq) == 0:
            continue
        counts = {b: seq.count(b) for b in set(seq)}
        ent = -sum((c / len(seq)) * math.log2(c / len(seq)) for c in counts.values() if c > 0)
        entropy_vals.append(ent)
    avg_entropy = round(float(np.mean(entropy_vals)), 3) if entropy_vals else 0.0

    # Class distribution
    class_dist: Dict[str, int] = {}
    for col in ["label", "mutation_type", "type", "consequence", "clinical_significance"]:
        if col in df.columns:
            class_dist = df[col].value_counts().to_dict()
            break

    # Reference availability
    has_ref = any(c in df.columns for c in ["reference", "ref_sequence", "reference_sequence", "ref"])

    return {
        "total_sequences": total_seqs,
        "valid_sequences": valid_seqs,
        "invalid_sequences": invalid_seqs,
        "duplicate_sequences": dup_count,
        "ambiguous_sequences": ambiguous_seqs,
        "mean_length": round(mean_len, 2),
        "min_length": min_len,
        "max_length": max_len,
        "gc_content": gc_frac,
        "mean_entropy": avg_entropy,
        "class_distribution": class_dist,
        "reference_availability": has_ref,
        "integrity_pass": (invalid_seqs == 0),
    }


def load_dataset(
    source: Union[str, Path, io.StringIO, io.BytesIO, pd.DataFrame],
    file_format: Optional[str] = None,
    sequence_col: Optional[str] = None,
    delimiter: Optional[str] = None,
    **kwargs: Any,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Master unified ingestion entry point.
    Loads and normalizes sequences from CSV, TSV, FASTA, FASTQ, or VCF,
    and returns (DataFrame, QualityAuditReport).
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
        df.columns = [str(c).strip().lower() for c in df.columns]
        audit = audit_data_quality(df, sequence_col=sequence_col or "sequence")
        return df, audit

    # Detect format if not provided
    fmt = (file_format or detect_file_format(source)).lower()
    logger.info(f"Ingesting dataset format '{fmt}'")

    content_str = ""
    if isinstance(source, (str, Path)) and os.path.exists(source):
        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            content_str = f.read()
    elif hasattr(source, "read"):
        raw = source.read()
        if isinstance(raw, bytes):
            content_str = raw.decode("utf-8", errors="ignore")
        else:
            content_str = str(raw)
    elif isinstance(source, str):
        content_str = source

    if fmt == "fasta":
        df = parse_fasta(content_str)
    elif fmt == "fastq":
        df = parse_fastq(content_str)
    elif fmt == "vcf":
        df = parse_vcf(content_str)
    else:  # CSV / TSV
        sep = delimiter or ("\t" if fmt == "tsv" else ",")
        # Read with StringIO
        try:
            df = pd.read_csv(io.StringIO(content_str), sep=sep)
        except Exception:
            # Fallback auto-detection
            df = pd.read_csv(io.StringIO(content_str), sep=None, engine="python")
        df.columns = [str(c).strip().lower() for c in df.columns]

    # Resolve standard sequence column name
    seq_candidate = sequence_col
    if not seq_candidate or seq_candidate not in df.columns:
        for c in ["sequence", "alt", "variant_seq", "dna_sequence", "seq"]:
            if c in df.columns:
                seq_candidate = c
                break

    if seq_candidate and seq_candidate != "sequence" and "sequence" not in df.columns:
        df["sequence"] = df[seq_candidate]
        seq_candidate = "sequence"

    audit = audit_data_quality(df, sequence_col=seq_candidate or "sequence")
    logger.info(
        f"Ingestion complete: {audit['total_sequences']} sequences, "
        f"Valid: {audit['valid_sequences']}, Invalid: {audit['invalid_sequences']}, "
        f"GC: {audit['gc_content']}%"
    )

    return df, audit

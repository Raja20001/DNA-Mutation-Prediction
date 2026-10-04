"""
DNA Sequence Cleaner and Dataset Sanitizer
Cleans raw DNA sequences, handles ambiguous bases, resolves missing values,
removes duplicates, applies length filters, and encodes labels reproducibly.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from .validator import (
    IUPAC_AMBIGUITY_CODES,
    STANDARD_NUCLEOTIDES,
    validate_sequence,
)


def clean_sequence(
    sequence: Any,
    uppercase: bool = True,
    strip_whitespace: bool = True,
) -> str:
    """
    Format and clean a DNA sequence string.

    Args:
        sequence: Raw input DNA sequence.
        uppercase: Whether to convert characters to uppercase.
        strip_whitespace: Whether to remove leading, trailing, and internal whitespace/newlines.

    Returns:
        str: Sanitized DNA string.
    """
    if sequence is None or pd.isna(sequence):
        return ""

    seq_str = str(sequence)
    if strip_whitespace:
        # Remove whitespace, tabs, and newlines (e.g. from FASTA formatting)
        seq_str = "".join(seq_str.split())

    if uppercase:
        seq_str = seq_str.upper()

    return seq_str


def handle_ambiguous_bases(
    sequence: str,
    action: str = "flag",
) -> Tuple[str, bool]:
    """
    Process IUPAC ambiguous bases in a DNA sequence.

    Args:
        sequence: The DNA sequence string (assumed uppercase).
        action: Strategy for handling ambiguous codes:
            - 'flag': keep characters as-is, return flag if ambiguity exists.
            - 'replace_n': replace any non-ACGT IUPAC character with 'N'.
            - 'filter': if any ambiguous base is present, marked for removal.

    Returns:
        Tuple[str, bool]:
            - processed_seq: Cleaned sequence according to strategy.
            - has_ambiguity: True if sequence contained any IUPAC ambiguity.
    """
    has_ambiguity = False
    ambig_keys = set(IUPAC_AMBIGUITY_CODES.keys())

    # Quick check
    chars_in_seq = set(sequence)
    if not chars_in_seq.intersection(ambig_keys):
        return sequence, False

    has_ambiguity = True

    if action == "flag":
        return sequence, True
    elif action == "replace_n":
        chars = [("N" if c in ambig_keys else c) for c in sequence]
        return "".join(chars), True
    elif action == "filter":
        return sequence, True
    else:
        raise ValueError(f"Unknown ambiguous action '{action}'. Choose from: 'flag', 'replace_n', 'filter'.")


class DNADataCleaner:
    """
    Orchestrates the comprehensive cleaning and sanitization of DNA sequence datasets.
    Maintains an audit trail of dropped records and the label encoding mapping.
    """

    def __init__(
        self,
        allowed_bases: Optional[Set[str]] = None,
        allow_ambiguous: bool = False,
        ambiguous_action: str = "flag",
        min_length: int = 10,
        max_length: int = 5000,
        remove_duplicates: bool = True,
        sequence_col: str = "sequence",
        label_col: str = "label",
    ):
        self.allowed_bases = allowed_bases or STANDARD_NUCLEOTIDES
        self.allow_ambiguous = allow_ambiguous
        self.ambiguous_action = ambiguous_action
        self.min_length = min_length
        self.max_length = max_length
        self.remove_duplicates = remove_duplicates
        self.sequence_col = sequence_col
        self.label_col = label_col

        self.label_encoder: Optional[LabelEncoder] = None
        self.label_mapping: Dict[Any, int] = {}
        self.audit_log: Dict[str, Any] = {}

    def clean_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute full cleaning pipeline on a DataFrame.

        Steps:
        1. Handle missing values in required columns (`sequence`, `label`).
        2. Format DNA sequences (uppercase, strip whitespace).
        3. Handle ambiguous bases according to strategy.
        4. Validate nucleotides against allowed set.
        5. Filter sequences by length constraints.
        6. Remove duplicate sequences (if enabled).
        7. Encode labels reproducibly.

        Args:
            df: Input DataFrame.

        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]:
                - cleaned_df: Cleaned and sanitized DataFrame.
                - audit_log: Detailed step-by-step audit metrics.
        """
        initial_records = len(df)
        working_df = df.copy()

        audit = {
            "initial_records": initial_records,
            "dropped_missing_sequence": 0,
            "dropped_missing_label": 0,
            "dropped_invalid_nucleotides": 0,
            "dropped_ambiguous": 0,
            "dropped_length_constraints": 0,
            "dropped_duplicates": 0,
            "final_records": 0,
            "label_mapping": {},
        }

        # Step 1: Drop missing sequence or label
        if self.sequence_col in working_df.columns:
            seq_missing_mask = working_df[self.sequence_col].isna() | (working_df[self.sequence_col].astype(str).str.strip() == "")
            audit["dropped_missing_sequence"] = int(seq_missing_mask.sum())
            working_df = working_df[~seq_missing_mask].copy()

        if self.label_col in working_df.columns:
            lbl_missing_mask = working_df[self.label_col].isna()
            audit["dropped_missing_label"] = int(lbl_missing_mask.sum())
            working_df = working_df[~lbl_missing_mask].copy()

        if len(working_df) == 0:
            audit["final_records"] = 0
            self.audit_log = audit
            return working_df, audit

        # Step 2: Clean sequence strings
        working_df[self.sequence_col] = working_df[self.sequence_col].apply(clean_sequence)

        # Step 3: Handle ambiguous bases
        processed_sequences = []
        ambiguity_flags = []
        drop_ambig_mask = []

        for seq in working_df[self.sequence_col]:
            p_seq, has_ambig = handle_ambiguous_bases(seq, action=self.ambiguous_action)
            processed_sequences.append(p_seq)
            ambiguity_flags.append(has_ambig)
            if self.ambiguous_action == "filter" and has_ambig:
                drop_ambig_mask.append(True)
            else:
                drop_ambig_mask.append(False)

        working_df[self.sequence_col] = processed_sequences
        working_df["has_ambiguity"] = ambiguity_flags

        if self.ambiguous_action == "filter":
            audit["dropped_ambiguous"] = sum(drop_ambig_mask)
            working_df = working_df[[not d for d in drop_ambig_mask]].copy()

        # Step 4: Validate nucleotides against allowed set
        valid_mask = []
        for seq in working_df[self.sequence_col]:
            is_valid, _, _ = validate_sequence(
                seq,
                allowed_bases=self.allowed_bases,
                allow_ambiguous=(self.allow_ambiguous or self.ambiguous_action != "filter"),
                min_length=1,
                max_length=100000,
            )
            valid_mask.append(is_valid)

        drop_invalid = sum(not v for v in valid_mask)
        audit["dropped_invalid_nucleotides"] = drop_invalid
        working_df = working_df[valid_mask].copy()

        # Step 5: Sequence length bounds
        lengths = working_df[self.sequence_col].str.len()
        length_mask = (lengths >= self.min_length) & (lengths <= self.max_length)
        audit["dropped_length_constraints"] = int((~length_mask).sum())
        working_df = working_df[length_mask].copy()

        # Step 6: Deduplication
        if self.remove_duplicates:
            # Check duplicate sequence
            before_dedup = len(working_df)
            working_df = working_df.drop_duplicates(subset=[self.sequence_col]).copy()
            audit["dropped_duplicates"] = before_dedup - len(working_df)

        # Step 7: Encode labels reproducibly
        if self.label_col in working_df.columns and len(working_df) > 0:
            self.label_encoder = LabelEncoder()
            # Convert to string to ensure stable sort ordering
            labels_raw = working_df[self.label_col].astype(str)
            encoded = self.label_encoder.fit_transform(labels_raw)
            working_df["label_encoded"] = encoded
            self.label_mapping = {
                str(cls_name): int(enc)
                for enc, cls_name in enumerate(self.label_encoder.classes_)
            }
            audit["label_mapping"] = self.label_mapping

        # Final audit tally
        working_df = working_df.reset_index(drop=True)
        audit["final_records"] = len(working_df)
        self.audit_log = audit

        return working_df, audit

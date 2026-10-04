"""
Evidence Table Module
Formats multi-source biological database evidence into a standardized tabular structure
for research evaluation and report generation.

Columns:
- Evidence Source
- Result
- Classification
- Condition
- Review Status
- Publication
- Population Frequency
- Evidence Notes
"""
from typing import Any, Dict, List, Optional
import pandas as pd


def generate_evidence_table(evidence_dict: Dict[str, Any]) -> pd.DataFrame:
    """
    Construct a standardized DataFrame containing evidence fields.

    Args:
        evidence_dict: Dictionary returned by BiologicalEvidenceConnector.

    Returns:
        pd.DataFrame formatted according to research specification.
    """
    row = {
        "Evidence Source": evidence_dict.get("source", "NCBI ClinVar / Ensembl"),
        "Result": "Found" if evidence_dict.get("found", False) else "Not Found",
        "Classification": evidence_dict.get("classification", "Not Classified"),
        "Condition": evidence_dict.get("condition", "N/A"),
        "Review Status": evidence_dict.get("review_status", "N/A"),
        "Publication": evidence_dict.get("publication", "N/A"),
        "Population Frequency": evidence_dict.get("population_frequency", "N/A"),
        "Evidence Notes": evidence_dict.get("evidence_notes", "N/A"),
    }
    return pd.DataFrame([row])


def format_evidence_markdown(evidence_dict: Dict[str, Any]) -> str:
    """
    Render formatted markdown table for evidence integration.
    """
    df = generate_evidence_table(evidence_dict)
    return df.to_markdown(index=False)

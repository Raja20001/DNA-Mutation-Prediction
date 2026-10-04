"""
Variant Normalization Layer
Normalizes genomic variants into standardized coordinate and allele notation
prior to external database annotation and clinical literature queries.

Captures:
- Reference genome assembly (e.g. GRCh38 / hg38)
- Chromosome
- Genomic coordinate (1-based closed)
- Reference allele
- Alternate allele
- Transcript identifier
- HGVS notation (cDNA / protein)
- Existing variant identifier (e.g. rsID)

CRITICAL RESEARCH RULE:
Explicitly distinguish between fields that were experimentally supplied vs. computationally derived.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Union
import pandas as pd


@dataclass
class NormalizedVariant:
    """Standardized genomic variant representation with provenance tracking."""

    assembly: str = "GRCh38"
    chromosome: str = "chrUnknown"
    position: Optional[int] = None
    reference_allele: str = "N"
    alternate_allele: str = "N"
    gene: str = "Unknown"
    transcript: str = "Unknown"
    hgvs_c: Optional[str] = None
    hgvs_p: Optional[str] = None
    variant_id: Optional[str] = None
    supplied_fields: Dict[str, Any] = field(default_factory=dict)
    derived_fields: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_variant(
    row_or_dict: Union[pd.Series, Dict[str, Any]],
    default_assembly: str = "GRCh38",
) -> NormalizedVariant:
    """
    Standardize raw variant metadata into a canonical NormalizedVariant object.

    Args:
        row_or_dict: Input row from dataset (Series or dictionary).
        default_assembly: Reference assembly ('GRCh38' or 'GRCh37').

    Returns:
        NormalizedVariant with provenance audit (supplied vs derived).
    """
    data = dict(row_or_dict)
    # Normalize keys to lowercase stripped strings
    clean_data = {str(k).strip().lower(): v for k, v in data.items()}

    supplied = {}
    derived = {}

    # 1. Chromosome normalization
    raw_chr = clean_data.get("chromosome") or clean_data.get("chr")
    if raw_chr and pd.notna(raw_chr):
        raw_chr_str = str(raw_chr).strip()
        chrom = raw_chr_str if raw_chr_str.startswith("chr") else f"chr{raw_chr_str}"
        supplied["chromosome"] = chrom
    else:
        chrom = "chrUnknown"
        derived["chromosome"] = chrom

    # 2. Position normalization
    raw_pos = clean_data.get("position") or clean_data.get("pos") or clean_data.get("start")
    if raw_pos is not None and pd.notna(raw_pos):
        try:
            pos = int(raw_pos)
            supplied["position"] = pos
        except (ValueError, TypeError):
            pos = None
            derived["position"] = "Invalid numeric position"
    else:
        pos = None
        derived["position"] = "Unavailable"

    # 3. Alleles
    raw_ref = clean_data.get("reference") or clean_data.get("ref")
    raw_alt = clean_data.get("alternate") or clean_data.get("alt")

    ref_allele = str(raw_ref).strip().upper() if (raw_ref and pd.notna(raw_ref)) else "N"
    alt_allele = str(raw_alt).strip().upper() if (raw_alt and pd.notna(raw_alt)) else "N"

    if raw_ref and pd.notna(raw_ref):
        supplied["reference_allele"] = ref_allele
    else:
        derived["reference_allele"] = ref_allele

    if raw_alt and pd.notna(raw_alt):
        supplied["alternate_allele"] = alt_allele
    else:
        derived["alternate_allele"] = alt_allele

    # 4. Gene and Transcript
    gene = str(clean_data.get("gene", "Unknown")).strip()
    if gene != "Unknown" and pd.notna(clean_data.get("gene")):
        supplied["gene"] = gene
    else:
        derived["gene"] = "Unknown"

    transcript = str(clean_data.get("transcript", "Unknown")).strip()
    if transcript != "Unknown" and pd.notna(clean_data.get("transcript")):
        supplied["transcript"] = transcript
    else:
        derived["transcript"] = "Unknown"

    # 5. HGVS & Variant Identifier
    raw_hgvs = clean_data.get("hgvs")
    if raw_hgvs and pd.notna(raw_hgvs):
        hgvs_str = str(raw_hgvs).strip()
        supplied["hgvs"] = hgvs_str
    else:
        # Derive canonical HGVS if transcript and position/alleles available
        if transcript != "Unknown" and pos is not None and ref_allele != "N" and alt_allele != "N":
            if len(ref_allele) == 1 and len(alt_allele) == 1:
                hgvs_str = f"{transcript}:c.{pos}{ref_allele}>{alt_allele}"
            else:
                hgvs_str = f"{transcript}:c.{pos}_{pos+len(ref_allele)-1}var"
            derived["hgvs"] = hgvs_str
        else:
            hgvs_str = None
            derived["hgvs"] = "Unavailable"

    variant_id = clean_data.get("variant_id") or clean_data.get("rsid") or clean_data.get("id")
    if variant_id and pd.notna(variant_id):
        supplied["variant_id"] = str(variant_id).strip()
    else:
        derived["variant_id"] = f"{chrom}_{pos}_{ref_allele}_{alt_allele}"

    return NormalizedVariant(
        assembly=default_assembly,
        chromosome=chrom,
        position=pos,
        reference_allele=ref_allele,
        alternate_allele=alt_allele,
        gene=gene,
        transcript=transcript,
        hgvs_c=hgvs_str,
        variant_id=str(supplied.get("variant_id", derived.get("variant_id"))),
        supplied_fields=supplied,
        derived_fields=derived,
    )

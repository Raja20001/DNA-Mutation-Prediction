"""
Unified Variant Data Model (VariantRecord)
Central standard object used across the entire DNA_v2 platform:
- Data Ingestion & Quality Audit
- Sequence Alignment & Extraction
- Normalization & Coordinate Handling
- Feature Engineering
- Multi-Model Predictions & Model Agreement
- Conformal Prediction & Uncertainty Quantification
- Genomic Annotation & Biological Interpretation
- ClinVar & External Evidence Provenance
- Explainability (XAI)
- Research Dossier & Web Dashboard
"""
from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
from typing import Any, Dict, List, Optional, Union


@dataclass
class VariantRecord:
    """
    Unified representation of a genomic variant and its full analytical lifecycle.
    """

    # Primary Identifiers & Genomic Coordinates
    variant_id: str = "VAR_UNKNOWN"
    sample_id: Optional[str] = None
    gene: str = "Unknown"
    chromosome: str = "chrUnknown"
    position: Optional[int] = None
    reference: str = "N"
    alternate: str = "N"

    # Mutation Typology & Subtype
    mutation_type: str = "Unknown"        # WILDTYPE, SNV, MNV, INSERTION, DELETION, DUPLICATION, DELINS, UNKNOWN
    mutation_subtype: str = "None"       # Transition, Transversion, In-frame, Frameshift, etc.

    # Sequences
    sequence_reference: Optional[str] = None
    sequence_alternate: Optional[str] = None

    # Normalization & Assembly
    assembly: str = "GRCh38"
    normalized_reference: Optional[str] = None
    normalized_alternate: Optional[str] = None

    # HGVS & Transcript Nomenclature
    hgvs_c: Optional[str] = None
    hgvs_p: Optional[str] = None
    transcript: str = "Unknown"
    consequence: str = "Unknown"

    # Primary / Champion Model Prediction
    prediction: str = "UNKNOWN"          # "MUTATION", "WILDTYPE", "UNCERTAIN"
    probability: float = 0.5
    confidence: str = "MEDIUM"           # "HIGH", "MEDIUM", "LOW", "UNCERTAIN"
    champion_model: str = "Ensemble"

    # Multi-Model Telemetry & Agreement
    model_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    model_agreement: float = 1.0         # Fraction (0.0 to 1.0) of models in agreement
    model_agreement_summary: str = "100% (1/1)"

    # Evidence & ClinVar Ground Truth
    evidence: Dict[str, Any] = field(default_factory=dict)
    evidence_source: str = "None"        # "Live ClinVar Result", "Cached ClinVar Result", "Local Benchmark Knowledge", "No Evidence Found"
    evidence_confidence: str = "None"    # Review status (e.g. "3 stars", "expert panel", "unreviewed")
    evidence_provenance: str = "derived" # "retrieved", "derived", "heuristic"

    # Explainability (XAI)
    explanation: Dict[str, Any] = field(default_factory=dict)
    top_features: List[Dict[str, Any]] = field(default_factory=list)

    # Uncertainty Quantification (Conformal Prediction)
    uncertainty: Dict[str, Any] = field(default_factory=dict)
    prediction_set: List[str] = field(default_factory=list)
    abstention: bool = False
    uncertainty_level: str = "LOW"       # "LOW", "MEDIUM", "HIGH"

    # Provenance Tracking (Supplied vs Derived)
    supplied_fields: Dict[str, Any] = field(default_factory=dict)
    derived_fields: Dict[str, Any] = field(default_factory=dict)

    # Timestamps & Metadata
    created_at: str = field(default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"))
    analysis_mode: str = "reference_aware"  # "reference_aware" or "reference_blind"
    status: str = "INITIALIZED"

    def to_dict(self) -> Dict[str, Any]:
        """Convert record to a standard dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VariantRecord":
        """Instantiate record safely from a dictionary, filtering unknown keys."""
        valid_fields = {f for f in cls.__dataclass_fields__.keys()}
        clean_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**clean_data)

    def to_json(self, indent: int = 2) -> str:
        """Serialize record to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "VariantRecord":
        """Deserialize record from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @property
    def is_mutated(self) -> bool:
        """Return True if positive mutation was predicted or detected."""
        return self.prediction.upper() in ["MUTATION", "YES", "MUTATED", "1"]

    @property
    def locus_display(self) -> str:
        """Formatted genomic locus string."""
        if self.chromosome and self.position:
            return f"{self.chromosome}:{self.position} ({self.assembly})"
        return "Unmapped coordinate"

    @property
    def allele_display(self) -> str:
        """Formatted allele change string (REF>ALT)."""
        ref = self.reference or "-"
        alt = self.alternate or "-"
        return f"{ref}>{alt}"

    @property
    def model_agreement_percentage(self) -> str:
        """Formatted agreement percentage string."""
        return f"{self.model_agreement * 100:.1f}%"

    @property
    def conformal_uncertainty_level(self) -> str:
        """Alias for uncertainty_level."""
        return self.uncertainty_level

    @property
    def conformal_abstention(self) -> bool:
        """Alias for abstention boolean."""
        return self.abstention

    @property
    def conformal_prediction_set(self) -> List[str]:
        """Alias for prediction_set."""
        return self.prediction_set



    def update_predictions(self, model_results: Dict[str, Dict[str, Any]], champion_key: Optional[str] = None) -> None:
        """
        Record multiple model predictions and calculate consensus agreement score.
        """
        self.model_predictions = model_results
        if not model_results:
            return

        votes = []
        for name, res in model_results.items():
            pred = res.get("prediction", res.get("mutation_detected", ""))
            is_mut = str(pred).upper() in ["MUTATION", "YES", "1", "TRUE"]
            votes.append(1 if is_mut else 0)

        total_models = len(votes)
        if total_models > 0:
            majority_class = 1 if (sum(votes) / total_models) >= 0.5 else 0
            agreement_count = sum(1 for v in votes if v == majority_class)
            self.model_agreement = round(agreement_count / total_models, 4)
            self.model_agreement_summary = f"{int(self.model_agreement * 100)}% ({agreement_count}/{total_models})"

        # Set champion
        if champion_key and champion_key in model_results:
            champ = model_results[champion_key]
            self.champion_model = champion_key
            self.probability = champ.get("probability", champ.get("mutation_probability", 0.5))
            self.prediction = "MUTATION" if self.probability >= 0.5 else "WILDTYPE"
        elif total_models > 0:
            avg_prob = sum(res.get("probability", 0.5) for res in model_results.values()) / total_models
            self.probability = round(avg_prob, 4)
            self.prediction = "MUTATION" if self.probability >= 0.5 else "WILDTYPE"

    def to_summary_dict(self) -> Dict[str, Any]:
        """Compact summary dictionary for APIs and telemetry."""
        return {
            "variant_id": self.variant_id,
            "sample_id": self.sample_id,
            "locus": f"{self.gene}:{self.chromosome}:{self.position or 'N/A'}",
            "mutation": f"{self.reference}>{self.alternate}",
            "mutation_type": self.mutation_type,
            "mutation_subtype": self.mutation_subtype,
            "consensus_prediction": self.prediction,
            "probability": self.probability,
            "confidence": self.confidence,
            "model_agreement": self.model_agreement_percentage,
            "conformal_uncertainty": self.conformal_uncertainty_level,
            "abstention": self.conformal_abstention,
            "evidence": self.evidence,
            "evidence_provenance": self.evidence_provenance,
        }

    def to_markdown_report(self) -> str:
        """Render complete structured 14-section research dossier."""
        lines = [
            "# DNA VARIANT RESEARCH DOSSIER",
            f"**Variant ID:** `{self.variant_id}` | **Generated:** `{self.created_at}`",
            "",
            "## 1. Executive Summary",
            f"- **Target Gene:** `{self.gene}`",
            f"- **Genomic Locus:** `{self.chromosome}:{self.position or 'N/A'}` ({self.assembly})",
            f"- **Mutation Event:** `{self.reference} → {self.alternate}` ({self.mutation_type} - {self.mutation_subtype})",
            f"- **Consensus Classification:** **{self.prediction}** (Probability: `{self.probability * 100:.1f}%`, Confidence: `{self.confidence}`)",
            f"- **Model Agreement:** `{self.model_agreement_percentage}`",
            f"- **Conformal Uncertainty:** `{self.conformal_uncertainty_level}` (Abstention: `{'YES' if self.conformal_abstention else 'NO'}`)",
            "",
            "## 2. Genomic Coordinates & Nomenclature",
            f"- **Transcript:** `{self.transcript}`",
            f"- **HGVS-c:** `{self.hgvs_c or 'N/A'}`",
            f"- **HGVS-p:** `{self.hgvs_p or 'N/A'}`",
            f"- **Predicted Consequence:** `{self.consequence}`",
            "",
            "## 3. Multi-Model Predictions & Consensus",
            "| Model | Call | Probability | Family |",
            "| :--- | :--- | :--- | :--- |",
        ]
        for m_name, m_info in self.model_predictions.items():
            prob = m_info.get("probability", 0.5)
            call = m_info.get("call", m_info.get("prediction", "UNKNOWN"))
            fam = m_info.get("family", "Classical")
            lines.append(f"| {m_name} | {call} | {prob * 100:.1f}% | {fam} |")

        lines.extend([
            "",
            "## 4. Evidence Grounding & Provenance",
            f"- **ClinVar Disease Association:** `{self.evidence or 'None reported'}`",
            f"- **Evidence Source:** `{self.evidence_source}`",
            f"- **Provenance Level:** `{self.evidence_provenance}`",
            f"- **Evidence Confidence:** `{self.evidence_confidence}`",
            "",
            "## 5. Mandatory Research Limitations",
            "- In-silico benchmark dataset; not validated on clinical patient cohorts.",
            "- Quantum computing models are evaluated on classical statevector simulators.",
            "- No claim of quantum computational supremacy or diagnostic clinical validity.",
            "- Consult certified molecular pathologists and clinicians for diagnostic purposes.",
            "",
            "---",
            "*DNA_v2 Hybrid Classical-Quantum Intelligence Platform*",
        ])
        return "\n".join(lines)


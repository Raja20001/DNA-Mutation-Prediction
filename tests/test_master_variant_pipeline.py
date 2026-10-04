"""
Tests for Master DNA Variant Intelligence Pipeline
Validates:
- Unified VariantRecord model
- Sequence Alignment (Needleman-Wunsch)
- Reference Manager (GRCh38, checksums, reference-blind mode)
- Unified Data Ingestion (FASTA, FASTQ, VCF, CSV)
- Model Ensemble and Agreement Scoring
- Conformal Uncertainty Quantification
- Master analyze_variant() pipeline execution
- Flask /api/analyze unified endpoint
"""
from pathlib import Path
import pytest
import pandas as pd

from src.alignment.aligner import align_sequences, is_transition
from src.ensemble.ensemble import EnsemblePredictor
from src.ingestion.loader import audit_data_quality, load_dataset, parse_fasta, parse_fastq, parse_vcf
from src.preprocessing.reference_manager import ReferenceManager
from src.uncertainty.conformal import ConformalPredictor
from src.variants.pipeline import analyze_variant
from src.variants.record import VariantRecord
from flask_app.app import app


class TestVariantRecord:
    def test_record_creation_and_serialization(self):
        rec = VariantRecord(
            variant_id="VAR_TEST_001",
            gene="TP53",
            chromosome="chr17",
            position=7577120,
            reference="C",
            alternate="T",
            mutation_type="SNV",
            mutation_subtype="Transition",
            prediction="MUTATION",
            probability=0.942,
            confidence="HIGH",
        )
        d = rec.to_dict()
        assert d["variant_id"] == "VAR_TEST_001"
        assert d["gene"] == "TP53"
        assert d["mutation_subtype"] == "Transition"

        summary = rec.to_summary_dict()
        assert summary["locus"] == "TP53:chr17:7577120"
        assert summary["consensus_prediction"] == "MUTATION"

        md = rec.to_markdown_report()
        assert "# DNA VARIANT RESEARCH DOSSIER" in md
        assert "TP53" in md


class TestSequenceAlignment:
    def test_needleman_wunsch_snv_alignment(self):
        ref = "ATCGATCG"
        alt = "ATCGTTCG"  # A -> T at pos 5
        res = align_sequences(ref, alt)
        assert res.edit_distance >= 1
        assert len(res.mutation_positions) == 1
        assert res.reference_base == "A"
        assert res.alternate_base == "T"
        assert res.identity_percent > 80.0
        assert is_transition("A", "G") is True
        assert is_transition("A", "T") is False

    def test_indel_alignment(self):
        ref = "ATCGATCG"
        alt = "ATCGGATCG"  # insertion of G
        res = align_sequences(ref, alt)
        assert res.edit_distance >= 1
        assert res.mutation_type in ["INSERTION", "DELINS", "DUPLICATION"] or any(
            str(getattr(op, "op_type", op.get("op_type") if isinstance(op, dict) else "")).lower() == "insertion"
            for op in res.edit_operations
        )




class TestReferenceManager:
    def test_reference_manager_audit_and_retrieval(self):
        rm = ReferenceManager()
        audit = rm.audit_reference()
        assert audit["reference_available"] is True
        assert audit["assembly"] == "GRCh38"

        ref_seq = rm.get_reference_sequence(gene="TP53")
        assert len(ref_seq) > 0
        assert set(ref_seq).issubset(set("ACGT"))


class TestUnifiedIngestion:
    def test_fasta_parsing(self):
        fasta_text = ">seq1 Sample A\nATGCTAGCTA\n>seq2 Sample B\nCGATCGATC"
        df = parse_fasta(fasta_text)
        assert len(df) == 2
        assert df.iloc[0]["sample_id"] == "seq1"
        assert df.iloc[0]["sequence"] == "ATGCTAGCTA"

    def test_fastq_parsing(self):
        fastq_text = "@READ_1\nATGC\n+\nIIII\n@READ_2\nCGTA\n+\nHHHH"
        df = parse_fastq(fastq_text)
        assert len(df) == 2
        assert df.iloc[0]["mean_phred_quality"] > 30.0

    def test_vcf_parsing(self):
        vcf_text = "##fileformat=VCFv4.2\n#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\nchr17\t7577120\trs123\tC\tT\t100\tPASS\t."
        df = parse_vcf(vcf_text)
        assert len(df) == 1
        assert df.iloc[0]["ref"] == "C"
        assert df.iloc[0]["alt"] == "T"

    def test_audit_data_quality(self):
        df = pd.DataFrame({
            "sequence": ["ATCGATCG", "ATCGNTCG", "INVALID123", "ATCGATCG"],
            "label": [0, 1, 1, 0],
        })
        audit = audit_data_quality(df)
        assert audit["total_sequences"] == 4
        assert audit["valid_sequences"] == 2
        assert audit["ambiguous_sequences"] == 1
        assert audit["invalid_sequences"] == 1
        assert audit["duplicate_sequences"] == 1


class TestEnsembleAndConformal:
    def test_ensemble_prediction_and_agreement(self):
        preds = {
            "Model_A": {"call": "MUTATION", "probability": 0.95, "family": "classical"},
            "Model_B": {"call": "MUTATION", "probability": 0.90, "family": "classical"},
            "Model_C": {"call": "MUTATION", "probability": 0.85, "family": "quantum"},
            "Model_D": {"call": "WILDTYPE", "probability": 0.30, "family": "deep_learning"},
        }
        ensemble = EnsemblePredictor(method="weighted_voting")
        res = ensemble.predict_from_dict(preds)
        assert res.prediction == "MUTATION"
        assert res.model_agreement >= 0.75
        assert res.model_agreement_summary == "3/4 models agree"
        assert res.is_disagreement_warning is False

    def test_conformal_uncertainty(self):
        cp = ConformalPredictor(alpha=0.10)
        cal_probs = [0.85, 0.92, 0.78, 0.95, 0.88, 0.80]
        cal_labels = [1, 1, 1, 1, 1, 1]
        cp.calibrate(cal_probs, cal_labels)

        res = cp.predict_with_uncertainty(0.94)
        assert res.prediction == "MUTATION"
        assert res.uncertainty_level in ["LOW", "MEDIUM"]
        assert "MUTATION" in res.prediction_set


class TestMasterAnalysisPipeline:
    def test_end_to_end_analyze_variant(self):
        ref = "ATCGATCGATCGATCGATCGATCG"
        alt = "ATCGATCGATCGTTCGATCGATCG"  # mutation
        rec = analyze_variant(
            sample_sequence=alt,
            reference_sequence=ref,
            gene="TP53",
            chromosome="chr17",
            position=7577120,
            analysis_mode="reference_aware",
        )
        assert isinstance(rec, VariantRecord)
        assert rec.gene == "TP53"
        assert rec.prediction in ["MUTATION", "WILDTYPE", "UNCERTAIN"]
        assert len(rec.model_predictions) > 0
        assert rec.evidence_provenance in ["retrieved", "cached", "local_benchmark", "heuristic", "derived", "not_found"]
        assert rec.conformal_uncertainty_level in ["LOW", "MEDIUM", "HIGH"]


class TestFlaskMasterEndpoint:
    @pytest.fixture
    def client(self):
        app.config["TESTING"] = True
        with app.test_client() as c:
            yield c

    def test_api_analyze_endpoint(self, client):
        ref = "ATCGATCGATCGATCGATCGATCG"
        alt = "ATCGATCGATCGTTCGATCGATCG"
        payload = {
            "sample_sequence": alt,
            "reference_sequence": ref,
            "gene": "TP53",
            "analysis_mode": "reference_aware",
        }
        res = client.post("/api/analyze", json=payload)
        assert res.status_code == 200
        data = res.get_json()
        assert data["status"].upper() == "SUCCESS"
        assert "variant" in data
        assert "predictions" in data
        assert "ensemble" in data
        assert "uncertainty" in data
        assert "evidence" in data


"""
Unit Tests for Disease Association Engine and Evidence Tables
"""
import pytest
from src.annotation.normalizer import normalize_variant
from src.disease_association.association import DiseaseAssociationEngine
from src.disease_association.connectors import BiologicalEvidenceConnector
from src.disease_association.evidence_table import generate_evidence_table


class TestDiseaseAssociation:
    def test_connector_known_gene(self):
        connector = BiologicalEvidenceConnector()
        res = connector.query_clinvar("BRAF")
        assert res["found"] is True
        assert "Melanoma" in res["condition"]
        assert "Pathogenic" in res["classification"]

    def test_disease_association_wildtype(self):
        var = normalize_variant({
            "gene": "BRAF",
            "reference": "A",
            "alternate": "A",
        })
        engine = DiseaseAssociationEngine()
        assoc = engine.evaluate_association(var)
        assert assoc["known_association"] == "NO"
        assert assoc["association_status"] == DiseaseAssociationEngine.NO_REPORTED_ASSOCIATION_FOUND

    def test_disease_association_reported(self):
        var = normalize_variant({
            "gene": "EGFR",
            "reference": "A",
            "alternate": "G",
            "position": 55242503,
        })
        engine = DiseaseAssociationEngine()
        assoc = engine.evaluate_association(var)
        assert assoc["known_association"] == "YES"
        assert assoc["association_status"] == DiseaseAssociationEngine.REPORTED_ASSOCIATION
        assert "patient" not in assoc["condition"].lower()  # Must not claim patient diagnosis

    def test_evidence_table_generation(self):
        sample_ev = {
            "source": "NCBI ClinVar",
            "found": True,
            "classification": "Pathogenic",
            "condition": "Colorectal Adenocarcinoma",
            "review_status": "criteria provided",
            "publication": "PMID:12345",
            "population_frequency": "0.0001",
            "evidence_notes": "Benchmark test note.",
        }
        df_ev = generate_evidence_table(sample_ev)
        assert len(df_ev) == 1
        assert "Evidence Source" in df_ev.columns
        assert df_ev.iloc[0]["Result"] == "Found"

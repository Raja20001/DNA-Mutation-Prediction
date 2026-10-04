"""
Unit Tests for DNA Variant Report Generator
"""
import pytest
from src.reporting.report_generator import DNAVariantReportGenerator


class TestReporting:
    def test_generate_report(self, tmp_path):
        gen = DNAVariantReportGenerator()

        seq = "ATGCCATGGAATGCCATGGA"
        detection = {"mutation_detected": "YES", "mutation_probability": 0.95, "model_name": "QMFN"}
        localization = {
            "position": 5,
            "reference_base": "C",
            "alternate_base": "T",
            "mutation_type": "substitution",
            "sequence_context": "ATGC[C>T]ATGGA",
        }
        annotation = {
            "assembly": "GRCh38",
            "chromosome": "chr7",
            "gene": "BRAF",
            "transcript": "ENST00000288602",
            "region": "Exon 15",
            "coding_status": "Coding",
            "molecular_consequence": "missense_variant",
            "codon_change": "GTG>GAG",
            "amino_acid_change": "p.Val600Glu",
        }
        interpretation = {
            "database_evidence": {
                "evolutionary_conservation": "High",
                "population_frequency": "0.0001",
                "protein_consequence": "Kinase activation",
                "biological_pathway": "MAPK",
                "evidence_source": "ClinVar",
            }
        }
        disease = {
            "known_association": "YES",
            "condition": "Melanoma",
            "evidence_source": "NCBI ClinVar",
            "classification": "Pathogenic",
            "review_status": "reviewed by expert panel",
            "literature": "PMID:12068308",
            "evidence_notes": "Activating mutation.",
        }

        save_file = tmp_path / "test_report.md"
        report_text = gen.generate_report(
            sequence=seq,
            detection_result=detection,
            localization_result=localization,
            annotation_result=annotation,
            interpretation_result=interpretation,
            disease_result=disease,
            save_path=save_file,
        )

        assert "DNA VARIANT ANALYSIS REPORT" in report_text
        assert "Mutation Detected:" in report_text
        assert "DISCLAIMER" in report_text
        assert save_file.exists()

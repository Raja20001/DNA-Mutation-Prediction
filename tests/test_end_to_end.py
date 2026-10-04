"""
End-to-End Pipeline Integration Test
Verifies complete flow on an in-silico synthetic benchmark dataset:
Data Ingestion -> Cleaning -> Splitting -> Feature Extraction -> QMFN Training ->
Mutation Detection -> Classification -> Localization -> Normalization ->
Annotation -> Biological Interpretation -> ClinVar Disease Query -> Report Generation.

SYNTHETIC DATA LABEL: All samples tested in this module are generated in-silico synthetically.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from src.annotation.annotator import GenomicAnnotator
from src.annotation.biological_interpretation import interpret_variant
from src.annotation.normalizer import normalize_variant
from src.disease_association.association import DiseaseAssociationEngine
from src.features.pipeline import run_feature_pipeline
from src.mutation.classifier import MutationClassifier
from src.mutation.detector import MutationDetector
from src.mutation.localizer import localize_mutation
from src.preprocessing.pipeline import DNAPreprocessingPipeline
from src.preprocessing.synthetic_data import generate_synthetic_dataset
from src.qmfnet.models import QMFNClassifier
from src.reporting.report_generator import DNAVariantReportGenerator


class TestEndToEndPipeline:
    def test_complete_research_pipeline(self, tmp_path):
        # 1. Generate small synthetic dataset
        df_synth = generate_synthetic_dataset(n_samples=50, seed=42)
        raw_csv = tmp_path / "synthetic_test_input.csv"
        df_synth.to_csv(raw_csv, index=False)

        custom_cfg = {
            "reproducibility": {"random_seed": 42},
            "data": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "test_size": 0.2,
                "val_size": 0.2,
                "min_sequence_length": 10,
                "max_sequence_length": 2000,
                "allowed_nucleotides": ["A", "C", "G", "T"],
                "ambiguous_action": "flag",
                "group_column": "gene",
            },
            "features": {
                "k_mer_sizes": [2, 3],
                "scale_features": True,
                "scaler_type": "standard",
            },
        }

        # 2. Preprocess
        preproc = DNAPreprocessingPipeline(config=custom_cfg)
        cleaned_df, train_df, val_df, test_df, prep_summary = preproc.run(
            input_source=raw_csv,
            save_artifacts=True,
        )
        assert len(cleaned_df) > 0
        assert prep_summary["initial_validation"]["validation_passed"] is True

        # 3. Extract Features
        X_train, X_val, X_test, extractor, feat_meta = run_feature_pipeline(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            config=custom_cfg,
            save_artifacts=True,
            output_dir=tmp_path / "processed",
        )
        assert len(X_train) == len(train_df)
        assert len(extractor.feature_names_) > 10

        y_train = train_df["label_encoded"].values
        y_test = test_df["label_encoded"].values

        # 4. Train proposed QMFN hybrid model
        qmfn = QMFNClassifier(
            classical_branch_dim=16,
            quantum_branch_qubits=2,
            circuit_depth=1,
            fusion_dim=8,
            epochs=4,
            batch_size=8,
            random_seed=42,
        )
        qmfn.fit(X_train, y_train)
        qmfn_metrics = qmfn.evaluate(X_test, y_test)
        assert qmfn_metrics["accuracy"] >= 0.0

        # 5. Downstream functional pipeline on test query sample
        sample_row = test_df.iloc[0]
        seq = sample_row["sequence"]

        # Detection
        detector = MutationDetector(model=qmfn, feature_extractor=extractor)
        det_res = detector.detect_sequence(seq, model_name="QMFN")
        assert det_res["mutation_detected"] in ["YES", "NO"]

        # Localization
        ref_seq = seq
        alt_seq = seq[:15] + "A" + seq[16:]
        loc_res = localize_mutation(ref_seq, alt_seq)
        assert loc_res["localized"] is True

        # Classification
        classifier = MutationClassifier()
        cls_res = classifier.classify_from_sequences("C", "A")
        assert cls_res["mutation_type"] == "Substitution"

        # Normalization & Annotation
        norm_var = normalize_variant(sample_row)
        annotator = GenomicAnnotator()
        annot_res = annotator.annotate(norm_var)
        assert annot_res["gene"] != ""

        # Disease Association
        disease_engine = DiseaseAssociationEngine()
        disease_res = disease_engine.evaluate_association(norm_var, custom_condition=sample_row.get("condition"))
        assert disease_res["known_association"] in ["YES", "NO", "UNCERTAIN"]

        # Biological Interpretation
        interp_res = interpret_variant(norm_var, prediction_result=det_res, external_evidence=disease_res)
        assert "model_prediction" in interp_res
        assert "database_evidence" in interp_res

        # 6. Report Generation
        report_gen = DNAVariantReportGenerator()
        report_file = tmp_path / "end_to_end_report.md"
        report_content = report_gen.generate_report(
            sequence=seq,
            detection_result=det_res,
            localization_result=loc_res,
            annotation_result=annot_res,
            interpretation_result=interp_res,
            disease_result=disease_res,
            save_path=report_file,
        )
        assert report_file.exists()
        assert "DNA VARIANT ANALYSIS REPORT" in report_content

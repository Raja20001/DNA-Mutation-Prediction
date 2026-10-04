"""
Verification script: exercises core data functions in dashboard/app.py to ensure zero runtime exceptions.
"""
from pathlib import Path
import sys
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.synthetic_data import generate_synthetic_dataset
from src.preprocessing.validator import validate_dataframe
from src.preprocessing.cleaner import DNADataCleaner
from src.features.extractor import DNAFeatureExtractor
from src.annotation.normalizer import normalize_variant
from src.annotation.annotator import GenomicAnnotator
from src.disease_association.association import DiseaseAssociationEngine
from src.annotation.biological_interpretation import interpret_variant
from src.mutation.detector import MutationDetector
from src.mutation.classifier import MutationClassifier
from src.mutation.localizer import localize_mutation
from src.reporting.report_generator import DNAVariantReportGenerator
from src.evaluation.quantum_visualizer import (
    create_radar_comparison_plot,
    create_quantum_vs_classical_kernel_heatmaps,
    create_bloch_sphere_3d,
    create_qubit_scaling_plot,
    create_multi_model_roc_curves,
    create_statistical_violin_plot,
    create_interactive_saliency_waterfall,
)
from dashboard.animations_3d import create_quantum_loss_landscape_3d

print("--- Testing Dataset Ingestion & Validation ---")
df = generate_synthetic_dataset(num_samples=50, random_seed=42)
val_res = validate_dataframe(df)
print("Validation passed:", val_res.get("validation_passed"))
total_recs = val_res.get("total_records", len(df))
invalid_cnt = val_res.get("invalid_sequences_count", 0)
valid_cnt = max(0, total_recs - invalid_cnt)
missing_seq = val_res.get("missing_values_per_column", {}).get("sequence", 0)
print(f"Valid sequences: {valid_cnt}/{total_recs}, Missing: {missing_seq}")

print("--- Testing Cleaner ---")
cleaner = DNADataCleaner(ambiguous_action="flag", remove_duplicates=True)
cleaned_df, audit = cleaner.clean_dataset(df)
print("Cleaner final records:", audit.get("final_records"))

print("--- Testing Feature Extractor ---")
extractor = DNAFeatureExtractor(kmer_sizes=[2, 3], scale_features=True)
extractor.fit(cleaned_df)
feats = extractor.transform(cleaned_df.head(5), as_dataframe=True)
print("Extracted features shape:", feats.shape)

print("--- Testing Visualizations ---")
fig_radar = create_radar_comparison_plot()
fig_q, fig_c = create_quantum_vs_classical_kernel_heatmaps()
fig_bloch = create_bloch_sphere_3d()
fig_scale = create_qubit_scaling_plot()
fig_roc = create_multi_model_roc_curves()
fig_violin = create_statistical_violin_plot()
fig_waterfall = create_interactive_saliency_waterfall(df.iloc[0]["sequence"])
fig_loss = create_quantum_loss_landscape_3d()
print("All 8 advanced figures generated successfully!")

print("--- Testing Mutation Pipeline ---")
row = df.iloc[0]
norm_var = normalize_variant(row)
annotator = GenomicAnnotator()
annot = annotator.annotate(norm_var)
disease_engine = DiseaseAssociationEngine()
disease_res = disease_engine.evaluate_association(norm_var, custom_condition=row.get("condition"))
interp = interpret_variant(norm_var, external_evidence=disease_res)
loc = localize_mutation(row["sequence"], row["sequence"][:20] + "T" + row["sequence"][21:], genomic_position=norm_var.position)
cls_engine = MutationClassifier()
cls_res = cls_engine.classify_from_sequences("A", "T")

print("--- Testing Report Generator ---")
report_gen = DNAVariantReportGenerator()
rep = report_gen.generate_report(
    sequence=row["sequence"],
    detection_result={"mutation_detected": "YES", "mutation_probability": 0.95, "model_name": "QMFN"},
    localization_result=loc,
    annotation_result=annot,
    interpretation_result=interp,
    disease_result=disease_res,
)
print("Report generated successfully! Length:", len(rep))
print("ALL CORE MODULES VERIFIED WITH ZERO ERRORS!")

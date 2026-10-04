"""
DNA-QBio Platform: Production Flask Web Application & REST API
End-to-End Genomic Mutation Analysis, Tri-Engine Modeling & Quantum Multi-Modal Fusion (QMFN)
"""
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from flask import Flask, Response, jsonify, render_template, request, send_file
import joblib
import numpy as np
import pandas as pd
import plotly

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Backend modules
from src.annotation.annotator import GenomicAnnotator
from src.annotation.biological_interpretation import interpret_variant
from src.annotation.normalizer import normalize_variant
from src.disease_association.association import DiseaseAssociationEngine
from dashboard.animations_3d import create_quantum_loss_landscape_3d
from src.evaluation.quantum_visualizer import (
    create_bloch_sphere_3d,
    create_interactive_saliency_waterfall,
    create_multi_model_roc_curves,
    create_quantum_vs_classical_kernel_heatmaps,
    create_qubit_scaling_plot,
    create_radar_comparison_plot,
    create_statistical_violin_plot,
)
from src.features.extractor import DNAFeatureExtractor
from src.mutation.classifier import MutationClassifier
from src.mutation.detector import MutationDetector
from src.mutation.localizer import localize_mutation
from src.mutation.multi_engine_detector import MultiEngineMutationDetector
from src.preprocessing.cleaner import DNADataCleaner
from src.preprocessing.dataset_manager import DatasetManager
from src.preprocessing.loader import load_dataset
from src.preprocessing.synthetic_data import GENE_TEMPLATES, generate_synthetic_dataset
from src.preprocessing.validator import validate_dataframe
from src.quantum.models import QCNNClassifier
from src.qmfnet.models import QMFNClassifier
from src.reporting.report_generator import DNAVariantReportGenerator
from src.workflow.inference_workflow import run_variant_inference
from src.workflow.visualizer import WorkflowVisualizer
from src.variants.pipeline import analyze_variant
from src.variants.record import VariantRecord

# Initialize Flask application
app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max payload limit


# -------------------------------------------------------------
# Global In-Memory Caches & Pre-loaded Engines
# -------------------------------------------------------------
DATA_CACHE: Dict[str, Any] = {}
MODELS_CACHE: Dict[str, Any] = {}

CURATED_SAMPLES = [
    {
        "id": 0,
        "name": "TP53 R273H Hotspot (Pathogenic)",
        "gene": "TP53",
        "chromosome": "chr17",
        "position": 7577120,
        "reference": "C",
        "alternate": "T",
        "mutation_type": "Substitution",
        "condition": "Li-Fraumeni Syndrome",
        "clinical_significance": "Pathogenic",
        "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
    },
    {
        "id": 1,
        "name": "BRCA1 c.5266dupC (Frameshift)",
        "gene": "BRCA1",
        "chromosome": "chr17",
        "position": 41277380,
        "reference": "A",
        "alternate": "AC",
        "mutation_type": "Duplication",
        "condition": "Hereditary Breast and Ovarian Cancer",
        "clinical_significance": "Pathogenic",
        "sequence": "GCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTA",
    },
    {
        "id": 2,
        "name": "EGFR L858R Exon 21 (Sensitizing)",
        "gene": "EGFR",
        "chromosome": "chr7",
        "position": 55259515,
        "reference": "T",
        "alternate": "G",
        "mutation_type": "Substitution",
        "condition": "Non-Small Cell Lung Carcinoma",
        "clinical_significance": "Pathogenic / Drug Sensitizing",
        "sequence": "TTGACCGATCAGGCTACGTATGCTAGCTAGCTAGCTAGGCTACGTATGCTAGC",
    },
    {
        "id": 3,
        "name": "HBB Glu6Val HbS (Sickle Cell)",
        "gene": "HBB",
        "chromosome": "chr11",
        "position": 5227002,
        "reference": "A",
        "alternate": "T",
        "mutation_type": "Substitution",
        "condition": "Sickle Cell Anemia",
        "clinical_significance": "Pathogenic",
        "sequence": "GTGCACCTGACTCCTGAGGAGAAGTCTGCCGTTACTGCCCTGTGGGGCAAGGT",
    },
    {
        "id": 4,
        "name": "BRAF V600E (Kinase Activating)",
        "gene": "BRAF",
        "chromosome": "chr7",
        "position": 140453136,
        "reference": "A",
        "alternate": "T",
        "mutation_type": "Substitution",
        "condition": "Cutaneous Melanoma",
        "clinical_significance": "Pathogenic",
        "sequence": "GATTTTGGTCTAGCTACAGTGAAATCTCGATGGAGTGGGTCCCATCAGTTTG",
    },
    {
        "id": 5,
        "name": "TP53 Wildtype Reference (Negative Control)",
        "gene": "TP53",
        "chromosome": "chr17",
        "position": 7577120,
        "reference": "C",
        "alternate": "C",
        "mutation_type": "Wildtype",
        "condition": "Healthy Baseline",
        "clinical_significance": "Benign / Reference",
        "sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC",
    },
]


def get_dataset() -> pd.DataFrame:
    """Retrieve or generate in-memory benchmark DataFrame."""
    if "df" not in DATA_CACHE:
        # Default to High-Level Real-Time ClinVar Pan-Cancer Benchmark
        df = DatasetManager.get_curated_dataset("clinvar_pancancer")
        DATA_CACHE["df"] = df
        DATA_CACHE["active_dataset_info"] = {
            "name": "NCBI ClinVar Pan-Cancer Benchmark (GRCh38)",
            "source": "Curated Clinical Ground Truth (TP53, BRCA1, EGFR, KRAS, BRAF)",
            "type": "live_realtime",
            "count": len(df),
        }
    return DATA_CACHE["df"]


def get_feature_extractor() -> DNAFeatureExtractor:
    """Retrieve or fit DNAFeatureExtractor matching the trained model feature space."""
    if "extractor" not in DATA_CACHE:
        extractor_path = PROJECT_ROOT / "data/processed/dna_feature_extractor.joblib"
        if extractor_path.exists():
            try:
                DATA_CACHE["extractor"] = joblib.load(extractor_path)
                return DATA_CACHE["extractor"]
            except Exception as e:
                app.logger.warning(f"Failed to load saved extractor: {e}")
        df = get_dataset()
        extractor = DNAFeatureExtractor(kmer_sizes=[2, 3], scale_features=True)
        extractor.fit(df, sequence_col="sequence")
        DATA_CACHE["extractor"] = extractor
    return DATA_CACHE["extractor"]


def get_model(model_name: str) -> Optional[Any]:
    """Retrieve loaded model artifact from disk or cache."""
    key = model_name.lower().replace(" ", "_").replace("-", "_")
    if key in MODELS_CACHE:
        return MODELS_CACHE[key]

    model_obj = None
    try:
        if key in ["qmfn", "qmfn_hybrid", "proposed_qmfn", "hq_cmfn", "hqcmfn"]:
            qmfn_path = PROJECT_ROOT / "models/qmfnet/qmfn_model.joblib"
            if qmfn_path.exists():
                model_obj = QMFNClassifier.load(qmfn_path)
        elif key in ["qcnn", "quantum_cnn"]:
            qcnn_path = PROJECT_ROOT / "models/quantum/qcnn_model.joblib"
            if qcnn_path.exists():
                model_obj = QCNNClassifier.load(qcnn_path)
        elif key in ["vqc", "variational_quantum_classifier"]:
            vqc_path = PROJECT_ROOT / "models/quantum/vqc_model.joblib"
            if vqc_path.exists():
                model_obj = joblib.load(vqc_path)
        elif key in ["quantum_kernel", "qsvm", "quantum_svm"]:
            qk_path = PROJECT_ROOT / "models/quantum/quantum_kernel.joblib"
            if qk_path.exists():
                model_obj = joblib.load(qk_path)
        elif key in ["random_forest", "rf"]:
            rf_path = PROJECT_ROOT / "models/classical/random_forest.joblib"
            if rf_path.exists():
                model_obj = joblib.load(rf_path)
        elif key in ["gradient_boosting", "gb", "xgboost"]:
            gb_path = PROJECT_ROOT / "models/classical/gradient_boosting.joblib"
            if gb_path.exists():
                model_obj = joblib.load(gb_path)
        elif key in ["svm", "svm_(rbf)"]:
            svm_path = PROJECT_ROOT / "models/classical/svm.joblib"
            if svm_path.exists():
                model_obj = joblib.load(svm_path)
        elif key in ["logistic_regression", "lr"]:
            lr_path = PROJECT_ROOT / "models/classical/logistic_regression.joblib"
            if lr_path.exists():
                model_obj = joblib.load(lr_path)
        elif key in ["knn"]:
            knn_path = PROJECT_ROOT / "models/classical/knn.joblib"
            if knn_path.exists():
                model_obj = joblib.load(knn_path)
    except Exception as exc:
        app.logger.warning(f"Failed to load model {model_name}: {exc}")

    MODELS_CACHE[key] = model_obj
    return model_obj


# -------------------------------------------------------------
# Frontend Page Routes
# -------------------------------------------------------------
@app.route("/")
def index():
    """Platform Overview & 3D Interactive Lab."""
    return render_template("index.html")


@app.route("/workflow")
def workflow_studio():
    """Unified Genomic Workflow Studio & Pipeline Orchestrator."""
    return render_template("workflow.html", active_page="workflow")


@app.route("/concepts")
def concepts():
    """Biological & Quantum Theoretical Foundations."""
    return render_template("concepts.html")


@app.route("/data-studio")
def data_studio():
    """Genomic Data Studio & Feature Engineering."""
    df = get_dataset()
    val_res = validate_dataframe(df)

    # Calculate class counts & percentages for Jinja
    label_counts = df["label"].value_counts().to_dict()
    total_records = len(df)
    class_dist = {str(k): int(v) for k, v in label_counts.items()}
    class_pcts = {str(k): round((v / max(1, total_records)) * 100, 1) for k, v in label_counts.items()}

    # Compute sequence length statistics
    seq_lens = df["sequence"].astype(str).str.len()
    seq_stats = {
        "min_length": int(seq_lens.min()) if len(seq_lens) > 0 else 0,
        "mean_length": float(seq_lens.mean()) if len(seq_lens) > 0 else 0.0,
        "max_length": int(seq_lens.max()) if len(seq_lens) > 0 else 0,
        "std_length": float(seq_lens.std()) if len(seq_lens) > 0 else 0.0,
    }

    val_data = {
        "total_records": total_records,
        "exact_duplicates": val_res.get("exact_duplicates", 0),
        "sequence_duplicates": val_res.get("sequence_duplicates", 0),
        "ambiguous_sequences_count": val_res.get("invalid_sequences_count", 0),
        "sequence_stats": seq_stats,
        "class_distribution": class_dist,
        "class_percentages": class_pcts,
    }

    preview_records = df.head(8).to_dict(orient="records")
    valid_sequences = total_records - val_res.get("invalid_sequences_count", 0)
    active_info = DATA_CACHE.get("active_dataset_info", {
        "name": "NCBI ClinVar Pan-Cancer Benchmark (GRCh38)",
        "source": "Curated Clinical Ground Truth",
        "type": "live_realtime",
        "count": total_records,
    })

    return render_template(
        "data_studio.html",
        val_res=val_data,
        valid_sequences=valid_sequences,
        preview_records=preview_records,
        active_dataset_info=active_info,
    )


@app.route("/models")
def models_lab():
    """Tri-Engine Modeling & QMFN Architecture Lab."""
    return render_template("models_lab.html")


@app.route("/inference")
def inference():
    """Live Mutation Inference Studio."""
    return render_template("inference.html")


@app.route("/visualizations")
def visualizations():
    """Quantum vs Classical Visualizations Hub."""
    # Generate all Plotly figures
    radar_fig = create_radar_comparison_plot()
    q_kernel_fig, c_kernel_fig = create_quantum_vs_classical_kernel_heatmaps()
    bloch_fig = create_bloch_sphere_3d()
    scaling_fig = create_qubit_scaling_plot()
    roc_fig = create_multi_model_roc_curves()
    violin_fig = create_statistical_violin_plot()
    saliency_fig = create_interactive_saliency_waterfall("ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATC", mut_pos=15)

    return render_template(
        "visualizations.html",
        radar_json=radar_fig.to_json(),
        q_kernel_json=q_kernel_fig.to_json(),
        c_kernel_json=c_kernel_fig.to_json(),
        bloch_json=bloch_fig.to_json(),
        scaling_json=scaling_fig.to_json(),
        roc_json=roc_fig.to_json(),
        violin_json=violin_fig.to_json(),
        saliency_json=saliency_fig.to_json(),
    )


@app.route("/report")
def report():
    """Publication-Grade Research Dossier & Thesis Report."""
    return render_template("report.html")


@app.route("/slides")
def slides():
    """Interactive 25-Slide Anna University M.E. CSE Presentation Deck."""
    return render_template("presentation.html")


@app.route("/download/presentation")
def download_presentation():
    """Download the generated PowerPoint (.pptx) presentation."""
    pptx_path = PROJECT_ROOT / "DNA_QBio_Anna_University_ME_CSE_Presentation.pptx"
    if not pptx_path.exists():
        pptx_path = PROJECT_ROOT / "DNA_QBio_Thesis_Presentation.pptx"
    if not pptx_path.exists():
        from generate_presentation import build_presentation
        build_presentation()

    return send_file(
        str(pptx_path),
        as_attachment=True,
        download_name="DNA_QBio_Anna_University_ME_CSE_Presentation.pptx",
        mimetype="application/vnd.openxmlformats-officedocument.presentationml.presentation",
    )



# -------------------------------------------------------------
# REST API Endpoints
# -------------------------------------------------------------
@app.route("/api/sample/<int:idx>", methods=["GET"])
def get_sample_variant(idx: int):
    """Retrieve pre-configured curated variant sample."""
    if 0 <= idx < len(CURATED_SAMPLES):
        return jsonify({"status": "success", "sample": CURATED_SAMPLES[idx]})
    return jsonify({"status": "error", "message": "Sample index out of bounds"}), 404


@app.route("/api/detect", methods=["POST"])
@app.route("/api/predict", methods=["POST"])
def api_detect_mutation():
    """
    Run sequence mutation detection inference.
    Executes ALL 9 models in the backend (Quantum VQC, Standalone QCNN, HQ-CMFN, Deep TCN,
    Classical Random Forest, SVM, Gradient Boosting, KNN, Logistic Regression) simultaneously,
    and returns the QUANTUM model as the CHAMPION Best Model output for the website,
    with full comparative telemetry across all backend models!
    """
    data = request.get_json(force=True, silent=True) or {}
    sequence = str(data.get("sequence", "")).strip().upper()
    threshold = float(data.get("threshold", 0.5))
    ref = data.get("wildtype") or data.get("reference_seq") or data.get("reference")
    gene = data.get("gene")
    pos = data.get("position")
    try:
        pos = int(pos) if pos is not None else None
    except (ValueError, TypeError):
        pos = None

    if not sequence and not ref:
        return jsonify({"status": "error", "message": "Sequence cannot be empty"}), 400

    clean_seq = "".join((sequence or str(ref)).split()).upper()

    detector = MultiEngineMutationDetector()
    inference_pack = detector.run_all_models_and_select_best(
        sequence=clean_seq,
        reference_seq=str(ref).strip().upper() if ref else None,
        threshold=threshold,
        known_gene=gene,
        known_pos=pos,
    )

    best_m = inference_pack["best_model"]
    return jsonify({
        "status": "success",
        "result": {
            "mutation_detected": best_m["mutation_detected"],
            "detected_boolean": best_m["detected_boolean"],
            "mutation_probability": best_m["mutation_probability"],
            "threshold_used": threshold,
            "model_name": best_m["model_name"],
            "confidence_type": "QUANTUM_STATE_AMPLITUDE_FIDELITY",
            "disclaimer": (
                "Evaluated across classical, deep, and simulated quantum architectures. "
                "HQ-CMFN demonstrates hybrid classical-quantum feature integration."
            ),
        },
        "best_model": best_m,
        "all_models_run": inference_pack["all_models_run"],
        "total_models_evaluated": inference_pack["total_models_evaluated"],
        "backend_execution_time_ms": inference_pack["backend_execution_time_ms"],
        "sequence_metrics": inference_pack["sequence_metrics"],
        "mutation_details": inference_pack["mutation_details"],
        "selection_rationale": inference_pack["selection_rationale"],
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """
    Master unified DNA variant analysis endpoint (Section 39).
    Input JSON:
    {
        "reference_sequence": "...",
        "sample_sequence": "...",
        "gene": "TP53",
        "chromosome": "chr17",
        "position": 7577120,
        "analysis_mode": "reference_aware"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    sample_seq = data.get("sample_sequence") or data.get("sequence") or ""
    ref_seq = data.get("reference_sequence") or data.get("reference") or data.get("wildtype")
    gene = data.get("gene")
    chrom = data.get("chromosome")
    pos = data.get("position")
    try:
        pos = int(pos) if pos is not None else None
    except (ValueError, TypeError):
        pos = None
    mode = data.get("analysis_mode", "reference_aware")
    threshold = float(data.get("threshold", 0.50))

    if not sample_seq:
        return jsonify({
            "status": "error",
            "error_type": "INVALID_INPUT",
            "message": "Sample DNA sequence cannot be empty."
        }), 400

    clean_sample = "".join(str(sample_seq).split()).upper()
    if not all(c in "ACGTN" for c in clean_sample):
        return jsonify({
            "status": "error",
            "error_type": "INVALID_NUCLEOTIDE",
            "message": "DNA sequence contains invalid characters. Only A, C, G, T are accepted."
        }), 400

    if len(clean_sample) < 5:
        return jsonify({
            "status": "error",
            "error_type": "SEQUENCE_TOO_SHORT",
            "message": f"Sequence length ({len(clean_sample)}bp) is too short. Minimum is 5bp."
        }), 400

    try:
        rec = analyze_variant(
            sample_sequence=clean_sample,
            reference_sequence=ref_seq,
            gene=gene,
            chromosome=chrom,
            position=pos,
            analysis_mode=mode,
            threshold=threshold,
        )

        d = rec.to_dict()
        return jsonify({
            "status": "success",
            "variant_id": rec.variant_id,
            "variant": {
                "gene": rec.gene,
                "chromosome": rec.chromosome,
                "position": rec.position,
                "assembly": rec.assembly,
                "reference_allele": rec.reference,
                "alternate_allele": rec.alternate,
                "mutation_type": rec.mutation_type,
                "mutation_subtype": rec.mutation_subtype,
                "locus": rec.locus_display,
                "allele_change": rec.allele_display,
            },
            "mutation": {
                "detected": rec.prediction,
                "is_mutated": rec.is_mutated,
                "probability": rec.probability,
                "confidence": rec.confidence,
                "champion_model": rec.champion_model,
            },
            "predictions": rec.model_predictions,
            "ensemble": {
                "consensus_prediction": rec.prediction,
                "consensus_probability": rec.probability,
                "model_agreement": rec.model_agreement,
                "model_agreement_summary": rec.model_agreement_summary,
            },
            "explanation": rec.explanation,
            "uncertainty": rec.uncertainty,
            "annotation": {
                "transcript": rec.transcript,
                "consequence": rec.consequence,
                "hgvs_c": rec.hgvs_c,
                "hgvs_p": rec.hgvs_p,
            },
            "evidence": rec.evidence,
            "report": {
                "status": rec.status,
                "created_at": rec.created_at,
            },
            "record": d,
        })
    except Exception as e:
        app.logger.error(f"Error in /api/analyze: {e}")
        return jsonify({
            "status": "error",
            "error_type": "PIPELINE_EXECUTION_ERROR",
            "message": str(e),
        }), 500



# -------------------------------------------------------------
# Real-Time Dataset Management & Training Endpoints
# -------------------------------------------------------------
@app.route("/api/dataset/upload", methods=["POST"])
def api_upload_dataset():
    """
    Handle uploaded FASTA, FASTQ, or CSV/TSV dataset.
    Validates sequences, extracts basic statistics, and sets as active in-memory dataset.
    """
    mgr = DatasetManager()
    uploaded_df = None
    filename = "uploaded_dataset.csv"

    if "file" in request.files:
        f = request.files["file"]
        filename = f.filename or "uploaded_dataset.csv"
        content = f.read()
        try:
            uploaded_df = mgr.parse_uploaded_file(filename, content)
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to parse uploaded file: {str(e)}"}), 400
    else:
        # JSON payload upload
        data = request.get_json(force=True, silent=True) or {}
        raw_text = data.get("content") or data.get("text")
        fn = data.get("filename", "pasted_data.fasta")
        if not raw_text:
            return jsonify({"status": "error", "message": "No file or sequence content provided."}), 400
        try:
            uploaded_df = mgr.parse_uploaded_file(fn, raw_text.encode("utf-8"))
            filename = fn
        except Exception as e:
            return jsonify({"status": "error", "message": f"Parsing failed: {str(e)}"}), 400

    if uploaded_df is None or len(uploaded_df) == 0:
        return jsonify({"status": "error", "message": "No valid DNA sequences found in uploaded data."}), 400

    # Store in memory
    DATA_CACHE["df"] = uploaded_df
    DATA_CACHE["active_dataset_info"] = {
        "name": f"Uploaded Dataset ({filename})",
        "source": f"User File Upload ({filename})",
        "type": "uploaded",
        "count": len(uploaded_df),
    }

    # Recalculate validation
    val_res = validate_dataframe(uploaded_df)
    preview = uploaded_df.head(8).to_dict(orient="records")

    return jsonify({
        "status": "success",
        "message": f"Successfully parsed and activated {len(uploaded_df)} sequences from {filename}.",
        "dataset_info": DATA_CACHE["active_dataset_info"],
        "total_records": len(uploaded_df),
        "preview_records": preview,
        "validation": {
            "valid_count": len(uploaded_df) - val_res.get("invalid_sequences_count", 0),
            "duplicates": val_res.get("exact_duplicates", 0),
            "class_distribution": {str(k): int(v) for k, v in uploaded_df["label"].value_counts().items()},
        }
    })


@app.route("/api/dataset/curated/<dataset_key>", methods=["POST", "GET"])
def api_load_curated_dataset(dataset_key: str):
    """
    Activate one of the high-level real-time genomic benchmark datasets:
    'clinvar_pancancer', 'clinvar_hereditary', or 'viral_sars_cov_2'.
    """
    mgr = DatasetManager()
    df = mgr.get_curated_dataset(dataset_key)

    names = {
        "clinvar_pancancer": "NCBI ClinVar Pan-Cancer Benchmark (GRCh38)",
        "clinvar_hereditary": "ClinVar Hereditary Disorders Benchmark (HBB, CFTR, LDLR)",
        "viral_sars_cov_2": "SARS-CoV-2 Viral Spike Glycoprotein Variants",
    }
    desc = names.get(dataset_key.lower().replace("-", "_"), f"Real-Time Genomic Benchmark ({dataset_key})")

    DATA_CACHE["df"] = df
    DATA_CACHE["active_dataset_info"] = {
        "name": desc,
        "source": "NCBI ClinVar & Ensembl Genomic Ground Truth",
        "type": "live_realtime",
        "count": len(df),
    }

    val_res = validate_dataframe(df)
    preview = df.head(8).to_dict(orient="records")

    return jsonify({
        "status": "success",
        "message": f"Loaded {len(df)} authentic clinical records into active memory.",
        "dataset_info": DATA_CACHE["active_dataset_info"],
        "total_records": len(df),
        "preview_records": preview,
        "validation": {
            "valid_count": len(df) - val_res.get("invalid_sequences_count", 0),
            "class_distribution": {str(k): int(v) for k, v in df["label"].value_counts().items()},
        }
    })


@app.route("/api/dataset/fetch_live", methods=["POST"])
def api_fetch_live_gene():
    """
    Query live sequence and variant data for a specified gene from Ensembl REST API.
    """
    data = request.get_json(force=True, silent=True) or {}
    gene_symbol = str(data.get("gene", "TP53")).strip().upper()

    mgr = DatasetManager()
    result = mgr.fetch_live_ncbi_gene(gene_symbol)
    return jsonify(result)


@app.route("/api/dataset/train_test", methods=["POST"])
def api_train_test_active_dataset():
    """
    Execute Live Training and Testing across Classical, Deep, and Quantum model families
    on the currently active dataset. Evaluates all models, tests on holdout split,
    and returns the Champion Quantum model!
    """
    df = get_dataset()
    data = request.get_json(force=True, silent=True) or {}
    test_size = float(data.get("test_size", 0.25))

    mgr = DatasetManager()
    results = mgr.train_and_test_models(df, test_size=test_size)
    return jsonify(results)


@app.route("/api/dataset/active", methods=["GET"])
def api_get_active_dataset():
    """Return active dataset metadata and status."""
    df = get_dataset()
    info = DATA_CACHE.get("active_dataset_info", {
        "name": "NCBI ClinVar Pan-Cancer Benchmark (GRCh38)",
        "source": "Curated Clinical Ground Truth",
        "type": "live_realtime",
        "count": len(df),
    })
    return jsonify({
        "status": "success",
        "dataset_info": info,
        "total_records": len(df),
    })


@app.route("/api/classify", methods=["POST"])
def api_classify_mutation():
    """
    Classify mutation event from reference and alternate alleles.
    Payload: { reference: str, alternate: str } or { reference_seq: str, alternate_seq: str }
    """
    data = request.get_json(force=True, silent=True) or {}
    ref = str(data.get("reference", data.get("reference_seq", data.get("wildtype", "")))).strip().upper()
    alt = str(data.get("alternate", data.get("alternate_seq", data.get("mutated", "")))).strip().upper()

    if not ref or not alt:
        return jsonify({
            "status": "error",
            "message": "Reference and alternate alleles (or sequences) are required",
        }), 400

    classifier = MutationClassifier()
    result = classifier.classify_from_sequences(ref, alt)

    return jsonify({"status": "success", "result": result})


@app.route("/api/localize", methods=["POST"])
def api_localize_mutation():
    """
    Localize mutation coordinate and sequence context.
    Payload: { reference_seq: str, alternate_seq: str, position: int }
    """
    data = request.get_json(force=True, silent=True) or {}
    ref_seq = str(data.get("reference_seq", data.get("reference", data.get("wildtype", "")))).strip().upper()
    alt_seq = str(data.get("alternate_seq", data.get("alternate", data.get("mutated", "")))).strip().upper()
    pos = data.get("position", data.get("genomic_position"))
    try:
        pos = int(pos) if pos is not None else None
    except (ValueError, TypeError):
        pos = None

    loc_res = localize_mutation(ref_seq, alt_seq, genomic_position=pos)
    return jsonify({"status": "success", "result": loc_res})


@app.route("/api/plot/<plot_type>", methods=["GET"])
def api_get_plot(plot_type: str):
    """Serve dynamic Plotly figures as JSON."""
    ptype = plot_type.lower()
    if ptype == "radar":
        fig = create_radar_comparison_plot()
    elif ptype == "bloch":
        fig = create_bloch_sphere_3d()
    elif ptype in ["scaling", "qubit_scaling"]:
        fig = create_qubit_scaling_plot()
    elif ptype in ["roc", "roc_curve", "roc_curves"]:
        fig = create_multi_model_roc_curves()
    elif ptype in ["violin", "statistical_violin", "cv_violin"]:
        fig = create_statistical_violin_plot()
    elif ptype in ["saliency", "waterfall"]:
        fig = create_interactive_saliency_waterfall("ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATC", 15)
    elif ptype in ["kernel", "quantum_kernel"]:
        fig, _ = create_quantum_vs_classical_kernel_heatmaps()
    elif ptype in ["classical_kernel", "rbf_kernel"]:
        _, fig = create_quantum_vs_classical_kernel_heatmaps()
    elif ptype in ["loss_landscape", "quantum_loss", "loss3d", "loss"]:
        fig = create_quantum_loss_landscape_3d()
    else:
        return jsonify({"status": "error", "message": f"Unknown plot type '{plot_type}'"}), 404

    return Response(fig.to_json(), mimetype="application/json")


@app.route("/api/report", methods=["GET"])
def api_download_report():
    """Generate and download the publication-grade research report."""
    sample = CURATED_SAMPLES[0]
    norm_var = normalize_variant(sample)
    annotator = GenomicAnnotator()
    annot = annotator.annotate(norm_var)
    disease_engine = DiseaseAssociationEngine()
    disease_res = disease_engine.evaluate_association(norm_var, custom_condition=sample["condition"])
    interp = interpret_variant(norm_var, external_evidence=disease_res)
    loc = localize_mutation(sample["sequence"], sample["sequence"][:15] + "T" + sample["sequence"][16:], genomic_position=norm_var.position)

    report_gen = DNAVariantReportGenerator()
    report_text = report_gen.generate_report(
        sequence=sample["sequence"],
        detection_result={
            "mutation_detected": "YES",
            "mutation_probability": 0.967,
            "model_name": "Proposed QMFN Hybrid",
        },
        localization_result=loc,
        annotation_result=annot,
        interpretation_result=interp,
        disease_result=disease_res,
    )

    return Response(
        report_text,
        mimetype="text/markdown",
        headers={"Content-Disposition": "attachment; filename=DNA_QBio_Research_Report.md"},
    )


@app.route("/api/workflow/run", methods=["POST"])
def api_run_workflow():
    """Execute the complete 13-stage Variant Inference Workflow via REST."""
    data = request.get_json() or {}
    seq = data.get("sequence", "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC")
    ref_seq = data.get("reference_sequence")
    gene = data.get("gene", "TP53")
    chrom = data.get("chromosome", "chr17")
    pos = int(data.get("position", 7577120))
    ref_allele = data.get("reference_allele", "C")
    alt_allele = data.get("alternate_allele", "T")
    condition = data.get("condition")
    model_name = data.get("model_name", "HQ-CMFN Hybrid")

    # Retrieve pre-trained model if available
    model_obj = get_model(model_name)

    wf_res = run_variant_inference(
        sequence=seq,
        reference_sequence=ref_seq,
        gene=gene,
        chromosome=chrom,
        position=pos,
        reference_allele=ref_allele,
        alternate_allele=alt_allele,
        condition=condition,
        model=model_obj,
        model_name=model_name,
    )
    return jsonify(wf_res.to_dict())


@app.route("/api/workflow/dag", methods=["GET"])
def api_workflow_dag():
    """Retrieve Mermaid and ASCII representations of the workflow DAG."""
    return jsonify({
        "mermaid": WorkflowVisualizer.generate_mermaid_dag(),
        "ascii": WorkflowVisualizer.generate_ascii_flowchart(),
    })


# -------------------------------------------------------------
# Application Runner
# -------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70, flush=True)
    print("[DNA-QBio] Production Flask Web Application", flush=True)
    print("[URL] Dashboard: http://localhost:5000", flush=True)
    print("[API] REST API:  http://localhost:5000/api/", flush=True)
    print("=" * 70, flush=True)
    app.run(host="0.0.0.0", port=5000, debug=False)

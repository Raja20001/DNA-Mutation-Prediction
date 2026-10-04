"""
DNA Variant Hybrid Framework: Master Training and Benchmarking Pipeline
Orchestrated via the Unified BatchBenchmarkWorkflow Engine:
1. Data Ingestion & Sanitization (Preprocessing Pipeline)
2. DNA Feature Extraction & Scaling
3. Classical ML Baselines (LR, RF, SVM, KNN, GB)
4. Deep Learning TCN Model
5. Quantum ML Baselines (VQC, Quantum Kernel, Standalone QCNN)
6. Proposed Hybrid HQ-CMFN / QMFN Architecture
7. Model Comparison & Statistical Significance Testing
8. Serialization of Models, Metrics, Figures, and Research Reports

Usage:
    python train.py                # Full benchmark execution
    python train.py --quick        # Fast benchmark execution (reduced epochs/samples)
    python train.py --skip-quantum # Fast execution skipping quantum simulation
"""
import argparse
from pathlib import Path
import time
import pandas as pd
import numpy as np

from src.utils.config import get_project_root, load_config
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.workflow.benchmark_workflow import BatchBenchmarkWorkflow
from src.workflow.inference_workflow import run_variant_inference
from src.workflow.base import WorkflowContext, WorkflowStatus
from src.workflow.visualizer import WorkflowVisualizer

logger = get_logger("train_pipeline")


def main():
    parser = argparse.ArgumentParser(description="Master Training Pipeline for DNA Variant Framework")
    parser.add_argument("--quick", action="store_true", help="Run with reduced iterations for fast validation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--skip-quantum", action="store_true", help="Skip quantum models for rapid testing")
    args = parser.parse_args()

    root = get_project_root()
    config = load_config(args.config)
    seed = config.get("reproducibility", {}).get("random_seed", 42)
    set_seed(seed)

    logger.info("=" * 70)
    logger.info("Starting DNA Variant Hybrid Classical-Quantum Master Pipeline")
    logger.info(f"Mode: {'QUICK_RUN' if args.quick else 'FULL_BENCHMARK'}, Seed: {seed}")
    logger.info("=" * 70)

    # 1. Initialize Observable Workflow Context
    ctx = WorkflowContext(
        config=config,
        initial_params={
            "quick_mode": args.quick,
            "skip_quantum": args.skip_quantum,
        },
    )

    # 2. Execute Batch Benchmark Workflow
    benchmark_workflow = BatchBenchmarkWorkflow()
    result = benchmark_workflow.execute(context=ctx)

    if result.status != WorkflowStatus.COMPLETED:
        logger.error(f"Benchmark workflow failed: {result.error}")
        return

    logger.info("\n" + result.to_markdown_summary())

    # 3. Generate Telemetry Waterfall Plot
    fig_dir = root / "results/figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    waterfall_path = fig_dir / "benchmark_workflow_waterfall.png"
    WorkflowVisualizer.plot_execution_waterfall(
        workflow_result=result,
        save_path=waterfall_path,
        title=f"Benchmark Workflow Execution Timeline ({'Quick' if args.quick else 'Full'})",
    )
    logger.info(f"Saved workflow telemetry waterfall to: {waterfall_path}")

    # 4. Synthesize Sample Variant Analysis Report using VariantInferenceWorkflow
    test_df = ctx.get("test_df")
    if test_df is not None and len(test_df) > 0:
        sample_row = test_df.iloc[0]
        logger.info(f"Generating clinical demonstration dossier for test sample ({sample_row.get('gene', 'Variant')})...")
        sample_res = run_variant_inference(
            sequence=sample_row["sequence"],
            gene=sample_row.get("gene", "TP53"),
            chromosome=sample_row.get("chromosome", "chr17"),
            position=sample_row.get("position", 7577120),
            reference_allele=sample_row.get("reference", "C"),
            alternate_allele=sample_row.get("alternate", "T"),
            condition=sample_row.get("condition", "Li-Fraumeni Syndrome"),
            config=config,
        )
        sample_report_path = root / "results/reports/sample_variant_report.md"
        if "variant_analysis_report" in sample_res.artifacts:
            import shutil
            shutil.copyfile(sample_res.artifacts["variant_analysis_report"], sample_report_path)
            logger.info(f"Sample Variant Report saved to: {sample_report_path}")

    logger.info("=" * 70)
    logger.info("MASTER TRAINING & BENCHMARKING PIPELINE SUCCESSFULLY COMPLETED!")
    logger.info(f"Duration: {result.duration_seconds}s")
    logger.info(f"Artifacts: {list(result.artifacts.keys())}")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()

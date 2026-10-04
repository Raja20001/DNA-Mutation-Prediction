"""
Unified Command-Line Interface for DNA Variant Framework Workflows.
Usage:
    # Run full variant inference workflow:
    python -m src.workflow.cli --mode inference --sequence "ATGCGATCGATCGATC" --gene "TP53"

    # Run batch benchmark training workflow:
    python -m src.workflow.cli --mode benchmark --quick

    # Print workflow DAG architecture:
    python -m src.workflow.cli --dag

    # List all registered workflow steps:
    python -m src.workflow.cli --list-steps
"""
import argparse
import json
import sys
from pathlib import Path

from .benchmark_workflow import run_batch_benchmark
from .inference_workflow import run_variant_inference
from .registry import StepRegistry
from .visualizer import WorkflowVisualizer
from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger

logger = get_logger("workflow_cli")


def main():
    parser = argparse.ArgumentParser(
        description="DNA-QBio Unified Workflow Engine CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode",
        choices=["inference", "benchmark"],
        default="inference",
        help="Workflow execution mode ('inference' for single/multi-variant analysis, 'benchmark' for batch training)",
    )
    parser.add_argument("--sequence", type=str, help="Query DNA sequence string for inference")
    parser.add_argument("--reference", type=str, help="Optional wildtype reference sequence for pairwise alignment")
    parser.add_argument("--gene", type=str, default="TP53", help="Gene symbol (e.g. BRAF, EGFR, KRAS, TP53)")
    parser.add_argument("--chromosome", type=str, default="chr17", help="Genomic chromosome identifier (e.g. chr7, chr17)")
    parser.add_argument("--position", type=int, default=7577120, help="1-based genomic coordinate")
    parser.add_argument("--ref-allele", type=str, default="C", help="Reference nucleotide allele")
    parser.add_argument("--alt-allele", type=str, default="T", help="Alternate nucleotide allele")
    parser.add_argument("--condition", type=str, help="Optional clinical condition / phenotypic context")
    parser.add_argument("--quick", action="store_true", help="Run in accelerated benchmark mode (reduced epochs/samples)")
    parser.add_argument("--skip-quantum", action="store_true", help="Skip quantum baseline training during batch benchmark")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dag", action="store_true", help="Display workflow architecture ASCII flowchart and Mermaid diagram")
    parser.add_argument("--list-steps", action="store_true", help="Print all registered workflow step definitions")

    args = parser.parse_args()

    if args.dag:
        print(WorkflowVisualizer.generate_ascii_flowchart())
        print("\n--- Mermaid DAG Syntax ---")
        print(WorkflowVisualizer.generate_mermaid_dag())
        return

    if args.list_steps:
        steps = StepRegistry.list_steps()
        print(f"\nRegistered Workflow Steps ({len(steps)} total):")
        for s in steps:
            print(f"  • {s}")
        return

    config = load_config(args.config)

    if args.mode == "benchmark":
        print("=" * 70)
        print("Launching Master Batch Benchmark Workflow...")
        print("=" * 70)
        res = run_batch_benchmark(
            quick=args.quick,
            skip_quantum=args.skip_quantum,
            config=config,
        )
        print("\n" + res.to_markdown_summary())
        print(f"\nBenchmark finished with status: {res.status.value} ({res.duration_seconds}s)")

    elif args.mode == "inference":
        # Default sample if none supplied
        seq = args.sequence or "ATGCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATCGATC"
        print("=" * 70)
        print("Launching Variant Inference Workflow...")
        print(f"Query Sequence: {seq[:30]}... (len={len(seq)}bp)")
        print(f"Gene Target:    {args.gene} ({args.chromosome}:{args.position})")
        print("=" * 70)

        res = run_variant_inference(
            sequence=seq,
            reference_sequence=args.reference,
            gene=args.gene,
            chromosome=args.chromosome,
            position=args.position,
            reference_allele=args.ref_allele,
            alternate_allele=args.alt_allele,
            condition=args.condition,
            config=config,
        )

        print("\n" + res.to_markdown_summary())
        print(f"\nInference finished with status: {res.status.value} ({res.duration_seconds}s)")
        if "variant_analysis_report" in res.artifacts:
            print(f"Report Generated: {res.artifacts['variant_analysis_report']}")


if __name__ == "__main__":
    main()

"""
DNA Variant Classical-Quantum Hybrid Framework: Unified Workflow Engine.
Provides modular, observable, reproducible pipeline orchestration for:
- VariantInferenceWorkflow: End-to-end single/multi-variant analysis from raw DNA to ClinVar & Reports.
- BatchBenchmarkWorkflow: Master training, evaluation, and statistical benchmarking across 9 architectures.
- WorkflowPipeline & WorkflowContext: Core abstractions for step-based execution, telemetry, and audit.
"""

from .base import (
    StepMetadata,
    StepResult,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from .registry import StepRegistry, register_step
from .orchestrator import WorkflowPipeline, WorkflowOrchestrator
from .inference_workflow import VariantInferenceWorkflow, run_variant_inference
from .benchmark_workflow import BatchBenchmarkWorkflow, run_batch_benchmark
from .visualizer import WorkflowVisualizer

__all__ = [
    "WorkflowStatus",
    "StepMetadata",
    "StepResult",
    "WorkflowContext",
    "WorkflowResult",
    "WorkflowStep",
    "StepRegistry",
    "register_step",
    "WorkflowPipeline",
    "WorkflowOrchestrator",
    "VariantInferenceWorkflow",
    "run_variant_inference",
    "BatchBenchmarkWorkflow",
    "run_batch_benchmark",
    "WorkflowVisualizer",
]

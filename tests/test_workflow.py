"""
Unit and Integration Tests for the Unified Workflow Engine.
Validates:
1. Core Workflow Abstractions (Context, StepResult, WorkflowResult, Status)
2. Step Registry & Dynamic Discovery
3. Pipeline Orchestrator Lifecycle, Hooks, Telemetry & Selective Execution
4. End-to-End VariantInferenceWorkflow (13-stage clinical variant analysis)
5. End-to-End BatchBenchmarkWorkflow (Batch dataset training & statistical benchmark)
6. Workflow Visualization Engine (Mermaid DAG, ASCII, Waterfall Chart)
"""
from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from src.workflow.base import (
    StepMetadata,
    StepResult,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from src.workflow.registry import StepRegistry, register_step
from src.workflow.orchestrator import WorkflowPipeline
from src.workflow.inference_workflow import (
    VariantInferenceWorkflow,
    run_variant_inference,
    SequenceIngestionStep,
    PreprocessingStep,
)
from src.workflow.benchmark_workflow import (
    BatchBenchmarkWorkflow,
    run_batch_benchmark,
)
from src.workflow.visualizer import WorkflowVisualizer


class DummySuccessStep(WorkflowStep):
    def __init__(self, name="DummyStep", stage_id=1):
        super().__init__(
            StepMetadata(
                name=name,
                stage_id=stage_id,
                category="Testing",
                description="Dummy step for testing lifecycle",
                expected_outputs=["test_val"],
            )
        )

    def execute(self, context: WorkflowContext) -> StepResult:
        res = StepResult(step_name=self.name, status=WorkflowStatus.RUNNING, stage_id=self.stage_id)
        context.set("test_val", 42)
        res.outputs = {"test_val": 42}
        res.finish(WorkflowStatus.COMPLETED)
        return res


class DummyFailingStep(WorkflowStep):
    def __init__(self, name="FailingStep", stage_id=2):
        super().__init__(
            StepMetadata(
                name=name,
                stage_id=stage_id,
                category="Testing",
                description="Dummy step that intentionally raises an error",
            )
        )

    def execute(self, context: WorkflowContext) -> StepResult:
        raise ValueError("Simulated step failure for unit test.")


class TestWorkflowCore:
    """Test fundamental workflow data structures, status transitions, and context operations."""

    def test_workflow_status_values(self):
        assert WorkflowStatus.PENDING.value == "PENDING"
        assert WorkflowStatus.RUNNING.value == "RUNNING"
        assert WorkflowStatus.COMPLETED.value == "COMPLETED"
        assert WorkflowStatus.SKIPPED.value == "SKIPPED"
        assert WorkflowStatus.FAILED.value == "FAILED"

    def test_step_result_lifecycle(self):
        res = StepResult(step_name="TestStep", status=WorkflowStatus.RUNNING, stage_id=1)
        assert res.status == WorkflowStatus.RUNNING
        assert res.duration_ms == 0.0

        res.finish(WorkflowStatus.COMPLETED)
        assert res.status == WorkflowStatus.COMPLETED
        assert res.duration_ms >= 0.0
        assert res.to_dict()["step_name"] == "TestStep"

    def test_workflow_context_set_get_has(self, tmp_path):
        ctx = WorkflowContext(initial_params={"seed": 42})
        assert ctx.get("seed") == 42
        assert ctx.has("seed") is True
        assert ctx.has("missing_key") is False

        ctx.set("gene", "BRAF")
        assert ctx.get("gene") == "BRAF"

        # Register artifact
        art_file = tmp_path / "test_artifact.txt"
        art_file.write_text("sample")
        ctx.register_artifact("sample_art", art_file)
        assert "sample_art" in ctx.artifacts

        summary = ctx.get_summary()
        assert "seed" in summary["data_keys"]
        assert summary["artifact_count"] == 1


class TestStepRegistry:
    """Test dynamic registration and lookup of workflow steps."""

    def test_step_registration(self):
        @register_step("mock_registered_step")
        class MockStep(WorkflowStep):
            def __init__(self):
                super().__init__(StepMetadata(name="Mock", stage_id=99, category="Test", description="Mock"))

            def execute(self, context: WorkflowContext) -> StepResult:
                res = StepResult(step_name=self.name, status=WorkflowStatus.COMPLETED)
                res.finish(WorkflowStatus.COMPLETED)
                return res

        assert "mock_registered_step" in StepRegistry.list_steps()
        step_instance = StepRegistry.instantiate("mock_registered_step")
        assert step_instance is not None
        assert step_instance.name == "Mock"


class TestWorkflowOrchestrator:
    """Test pipeline execution sequencing, hooks, and failure policies."""

    def test_pipeline_execution_success(self):
        pipe = WorkflowPipeline(name="TestPipe")
        pipe.add_step(DummySuccessStep())

        hook_called = []
        pipe.add_pre_step_hook(lambda s, c: hook_called.append(s.name))

        result = pipe.execute()
        assert result.is_successful is True
        assert len(result.step_results) == 1
        assert result.step_results[0].outputs["test_val"] == 42
        assert len(hook_called) == 1

    def test_pipeline_fail_fast(self):
        pipe = WorkflowPipeline(name="FailFastPipe", fail_fast=True)
        pipe.add_step(DummyFailingStep())
        pipe.add_step(DummySuccessStep())

        result = pipe.execute()
        assert result.status == WorkflowStatus.FAILED
        assert "Simulated step failure" in result.error
        # Second step must not have been executed because fail_fast is True
        assert len(result.step_results) == 1

    def test_selective_stage_execution(self):
        pipe = WorkflowPipeline(name="SelectivePipe")
        pipe.add_step(DummySuccessStep("StepA", stage_id=1))
        pipe.add_step(DummySuccessStep("StepB", stage_id=2))
        pipe.add_step(DummySuccessStep("StepC", stage_id=3))

        result = pipe.execute(stages=[1, 3])
        assert len(result.step_results) == 2
        names = [r.step_name for r in result.step_results]
        assert "StepA" in names
        assert "StepC" in names
        assert "StepB" not in names


class TestVariantInferenceWorkflow:
    """Integration test of the complete 13-stage VariantInferenceWorkflow."""

    def test_full_variant_inference_execution(self, tmp_path):
        seq = "ATGCGATCGATCGATCGATCGATCGATCGATC"
        result = run_variant_inference(
            sequence=seq,
            gene="TP53",
            chromosome="chr17",
            position=7577120,
            reference_allele="C",
            alternate_allele="T",
            condition="Li-Fraumeni Syndrome",
        )

        assert result.is_successful is True
        assert len(result.step_results) == 13

        # Check critical stage outputs
        summary = result.summary_data
        assert summary.get("mutation_detected") in ["YES", "NO"]
        assert "variant_analysis_report" in result.artifacts

        # Verify report generation on disk
        report_file = Path(result.artifacts["variant_analysis_report"])
        assert report_file.exists()
        content = report_file.read_text(encoding="utf-8")
        assert "DNA VARIANT ANALYSIS REPORT" in content
        assert "TP53" in content


class TestBatchBenchmarkWorkflow:
    """Integration test of the 8-stage BatchBenchmarkWorkflow in quick mode."""

    def test_quick_batch_benchmark(self):
        result = run_batch_benchmark(quick=True, skip_quantum=True)
        assert result.is_successful is True
        assert len(result.step_results) == 8

        # Quantum baselines should be skipped when skip_quantum=True
        q_step = result.get_step("Quantum ML Baselines (VQC, Quantum Kernel, Standalone QCNN)")
        assert q_step is not None
        assert q_step.status == WorkflowStatus.SKIPPED

        # Verify models evaluated
        classical_step = result.get_step("Classical ML Benchmarking (LR, RF, SVM, KNN, GB)")
        assert classical_step is not None
        assert classical_step.status == WorkflowStatus.COMPLETED
        assert classical_step.outputs["models_evaluated"] >= 4


class TestWorkflowVisualizer:
    """Test diagram generation and waterfall telemetry plotting."""

    def test_dag_and_flowchart_generation(self):
        dag = WorkflowVisualizer.generate_mermaid_dag()
        assert "graph TD" in dag
        assert "Sequence Ingestion" in dag
        assert "ClinVar" in dag

        ascii_chart = WorkflowVisualizer.generate_ascii_flowchart()
        assert "DNA-QBio" in ascii_chart
        assert "Ingestion" in ascii_chart

    def test_waterfall_plot_rendering(self, tmp_path):
        pipe = WorkflowPipeline(name="PlotTest")
        pipe.add_step(DummySuccessStep("Step1", stage_id=1))
        pipe.add_step(DummySuccessStep("Step2", stage_id=2))
        res = pipe.execute()

        plot_file = tmp_path / "test_waterfall.png"
        WorkflowVisualizer.plot_execution_waterfall(res, save_path=plot_file)
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0

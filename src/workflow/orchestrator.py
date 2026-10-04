"""
Workflow Orchestrator and Pipeline Engine.
Manages step sequencing, dependency checking, telemetry collection,
and structured execution of research workflows.
"""
from datetime import datetime
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Union

import pandas as pd

from .base import (
    StepMetadata,
    StepResult,
    WorkflowContext,
    WorkflowResult,
    WorkflowStatus,
    WorkflowStep,
)
from ..utils.logger import get_logger

logger = get_logger("workflow_orchestrator")


class WorkflowPipeline:
    """
    Sequences and executes a list of WorkflowStep instances inside a WorkflowContext.
    Provides execution lifecycle hooks, telemetry, and selective stage filtering.
    """

    def __init__(
        self,
        name: str = "DNAVariantWorkflow",
        description: str = "Modular DNA Variant Research Workflow",
        fail_fast: bool = True,
    ):
        self.name = name
        self.description = description
        self.fail_fast = fail_fast
        self.steps: List[WorkflowStep] = []
        self._pre_step_hooks: List[Callable[[WorkflowStep, WorkflowContext], None]] = []
        self._post_step_hooks: List[Callable[[WorkflowStep, StepResult, WorkflowContext], None]] = []

    def add_step(self, step: WorkflowStep) -> "WorkflowPipeline":
        """Add a workflow step to the sequence."""
        self.steps.append(step)
        return self

    def add_pre_step_hook(self, hook: Callable[[WorkflowStep, WorkflowContext], None]):
        """Register a callback executed immediately prior to each step."""
        self._pre_step_hooks.append(hook)

    def add_post_step_hook(self, hook: Callable[[WorkflowStep, StepResult, WorkflowContext], None]):
        """Register a callback executed immediately after each step completes."""
        self._post_step_hooks.append(hook)

    def get_step_names(self) -> List[str]:
        """List names of all queued steps."""
        return [s.name for s in self.steps]

    def execute(
        self,
        context: Optional[WorkflowContext] = None,
        stages: Optional[List[Union[int, str]]] = None,
        progress_callback: Optional[Callable[[int, int, str, StepResult], None]] = None,
    ) -> WorkflowResult:
        """
        Execute the pipeline.

        Args:
            context: Shared WorkflowContext. If None, a new context is initialized.
            stages: Optional filter of stage IDs or step names to run.
            progress_callback: Optional callback (current_idx, total_steps, step_name, result).

        Returns:
            WorkflowResult containing full execution details and telemetry.
        """
        ctx = context or WorkflowContext()
        start_wall_time = time.time()
        start_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

        logger.info(f"=== [START WORKFLOW] {self.name} ===")
        logger.info(f"Configured Steps: {len(self.steps)} | Fail-Fast: {self.fail_fast}")

        # Filter steps if requested
        steps_to_run: List[WorkflowStep] = []
        for step in self.steps:
            if stages is None:
                steps_to_run.append(step)
            else:
                stage_match = (step.stage_id in stages) or (step.name.lower() in [str(s).lower() for s in stages])
                if stage_match:
                    steps_to_run.append(step)

        logger.info(f"Active Steps to Execute: {len(steps_to_run)}")

        step_results: List[StepResult] = []
        pipeline_status = WorkflowStatus.COMPLETED
        pipeline_error: Optional[str] = None

        total_steps = len(steps_to_run)
        for idx, step in enumerate(steps_to_run):
            step_name = step.name
            stage_id = step.stage_id

            logger.info(f"--> [Stage {stage_id}] Running: {step_name} ({idx + 1}/{total_steps})")

            # Check skip condition
            can_skip, skip_reason = step.can_skip(ctx)
            if can_skip:
                logger.info(f"Skipping {step_name}: {skip_reason}")
                res = StepResult(
                    step_name=step_name,
                    status=WorkflowStatus.SKIPPED,
                    stage_id=stage_id,
                    skipped_reason=skip_reason,
                )
                res.finish(WorkflowStatus.SKIPPED)
                step_results.append(res)
                ctx.add_step_result(res)
                if progress_callback:
                    progress_callback(idx + 1, total_steps, step_name, res)
                continue

            # Validate input dependencies
            valid_inputs, missing = step.validate_inputs(ctx)
            if not valid_inputs:
                err_msg = f"Step '{step_name}' missing required inputs in context: {missing}"
                logger.error(err_msg)
                res = StepResult(
                    step_name=step_name,
                    status=WorkflowStatus.FAILED,
                    stage_id=stage_id,
                    error=err_msg,
                )
                res.finish(WorkflowStatus.FAILED, error=err_msg)
                step_results.append(res)
                ctx.add_step_result(res)

                if progress_callback:
                    progress_callback(idx + 1, total_steps, step_name, res)

                if self.fail_fast:
                    pipeline_status = WorkflowStatus.FAILED
                    pipeline_error = err_msg
                    break
                else:
                    continue

            # Pre-step hooks
            for hook in self._pre_step_hooks:
                try:
                    hook(step, ctx)
                except Exception as e:
                    logger.warning(f"Pre-step hook error on {step_name}: {e}")

            # Execute step
            try:
                res = step.execute(ctx)
                if res.status == WorkflowStatus.RUNNING:
                    res.finish(WorkflowStatus.COMPLETED)
            except Exception as e:
                import traceback
                err_trace = traceback.format_exc()
                err_str = f"Execution exception in step '{step_name}': {str(e)}"
                logger.error(f"{err_str}\n{err_trace}")

                res = StepResult(
                    step_name=step_name,
                    status=WorkflowStatus.FAILED,
                    stage_id=stage_id,
                    error=err_str,
                    logs=[err_trace],
                )
                res.finish(WorkflowStatus.FAILED, error=err_str)

                step_results.append(res)
                ctx.add_step_result(res)

                if progress_callback:
                    progress_callback(idx + 1, total_steps, step_name, res)

                if self.fail_fast:
                    pipeline_status = WorkflowStatus.FAILED
                    pipeline_error = err_str
                    break
                else:
                    continue

            step_results.append(res)
            ctx.add_step_result(res)

            # Post-step hooks
            for hook in self._post_step_hooks:
                try:
                    hook(step, res, ctx)
                except Exception as e:
                    logger.warning(f"Post-step hook error on {step_name}: {e}")

            if progress_callback:
                progress_callback(idx + 1, total_steps, step_name, res)

            if res.status == WorkflowStatus.FAILED and self.fail_fast:
                pipeline_status = WorkflowStatus.FAILED
                pipeline_error = res.error or f"Step {step_name} failed"
                break

        end_wall_time = time.time()
        end_time_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        total_duration = round(end_wall_time - start_wall_time, 3)

        logger.info(
            f"=== [WORKFLOW FINISHED] Status: {pipeline_status.value} "
            f"in {total_duration}s ==="
        )

        # Build summary outputs
        summary_data: Dict[str, Any] = {}
        for key in ["mutation_detected", "mutation_type", "position", "gene", "condition", "known_association"]:
            if ctx.has(key):
                summary_data[key] = ctx.get(key)

        return WorkflowResult(
            pipeline_name=self.name,
            status=pipeline_status,
            start_time=start_time_str,
            end_time=end_time_str,
            duration_seconds=total_duration,
            step_results=step_results,
            artifacts={k: str(v) for k, v in ctx.artifacts.items()},
            summary_data=summary_data,
            error=pipeline_error,
        )


# Alias for backwards compatibility
WorkflowOrchestrator = WorkflowPipeline

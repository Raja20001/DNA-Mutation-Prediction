"""
Core Abstractions for the DNA Variant Workflow Engine.
Defines base interfaces, status enums, execution context, step results, and telemetry tracking.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json
from pathlib import Path
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger

logger = get_logger("workflow_base")


class WorkflowStatus(str, Enum):
    """Execution status for workflow pipelines and individual steps."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass
class StepMetadata:
    """Metadata detailing a workflow step's requirements, intent, and outputs."""
    name: str
    stage_id: int
    category: str  # e.g., "Ingestion", "Preprocessing", "Modeling", "Annotation", "Evidence", "XAI", "Reporting"
    description: str
    required_inputs: List[str] = field(default_factory=list)
    optional_inputs: List[str] = field(default_factory=list)
    expected_outputs: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """Result and telemetry of an individual workflow step execution."""
    step_name: str
    status: WorkflowStatus
    stage_id: int = 0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    duration_ms: float = 0.0
    outputs: Dict[str, Any] = field(default_factory=dict)
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    skipped_reason: Optional[str] = None

    def finish(self, status: WorkflowStatus = WorkflowStatus.COMPLETED, error: Optional[str] = None):
        """Finalize the step timing and status."""
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000.0, 2)
        self.status = status
        if error:
            self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert step result to a JSON-serializable dictionary."""
        return {
            "step_name": self.step_name,
            "stage_id": self.stage_id,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "output_keys": list(self.outputs.keys()),
            "logs_count": len(self.logs),
            "error": self.error,
            "skipped_reason": self.skipped_reason,
        }


class WorkflowContext:
    """
    Observable, thread-safe execution context for pipeline workflows.
    Maintains shared state, artifacts registry, configuration parameters, and telemetry.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "config.yaml",
        initial_params: Optional[Dict[str, Any]] = None,
    ):
        self.config: Dict[str, Any] = config or load_config(config_path)
        self.root: Path = get_project_root()
        self.data: Dict[str, Any] = initial_params.copy() if initial_params else {}
        self.artifacts: Dict[str, Path] = {}
        self.step_history: List[StepResult] = []
        self.listeners: List[Callable[[str, StepResult], None]] = []
        self.created_at: str = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    def set(self, key: str, value: Any):
        """Store a value into the workflow context."""
        self.data[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """Retrieve a value from the workflow context."""
        return self.data.get(key, default)

    def has(self, key: str) -> bool:
        """Check if a key exists in context."""
        return key in self.data

    def register_artifact(self, name: str, path: Union[str, Path]):
        """Register an output file artifact into the context."""
        self.artifacts[name] = Path(path)

    def add_step_result(self, result: StepResult):
        """Append a step result and notify listeners."""
        self.step_history.append(result)
        for listener in self.listeners:
            try:
                listener(result.step_name, result)
            except Exception as e:
                logger.warning(f"Workflow listener error on step {result.step_name}: {e}")

    def add_listener(self, callback: Callable[[str, StepResult], None]):
        """Attach a subscriber for real-time step progress updates."""
        self.listeners.append(callback)

    def get_summary(self) -> Dict[str, Any]:
        """Generate a summary of stored data keys and execution history."""
        return {
            "created_at": self.created_at,
            "data_keys": list(self.data.keys()),
            "artifact_count": len(self.artifacts),
            "steps_executed": [res.to_dict() for res in self.step_history],
        }


class WorkflowStep(ABC):
    """
    Abstract Base Class for an individual, modular step in a workflow pipeline.
    """

    def __init__(self, metadata: StepMetadata):
        self.metadata = metadata

    @property
    def name(self) -> str:
        return self.metadata.name

    @property
    def stage_id(self) -> int:
        return self.metadata.stage_id

    def validate_inputs(self, context: WorkflowContext) -> Tuple[bool, List[str]]:
        """Verify that all required inputs exist in the context."""
        missing = [inp for inp in self.metadata.required_inputs if not context.has(inp)]
        return len(missing) == 0, missing

    def can_skip(self, context: WorkflowContext) -> Tuple[bool, str]:
        """Determine if this step can be safely skipped based on context."""
        return False, ""

    @abstractmethod
    def execute(self, context: WorkflowContext) -> StepResult:
        """
        Execute step logic, update context with outputs, and return StepResult.
        """
        pass


@dataclass
class WorkflowResult:
    """Consolidated outcome of a full workflow pipeline run."""
    pipeline_name: str
    status: WorkflowStatus
    start_time: str
    end_time: str
    duration_seconds: float
    step_results: List[StepResult] = field(default_factory=list)
    artifacts: Dict[str, str] = field(default_factory=dict)
    summary_data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.status == WorkflowStatus.COMPLETED

    def get_step(self, step_name: str) -> Optional[StepResult]:
        for res in self.step_results:
            if res.step_name.lower() == step_name.lower():
                return res
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert full workflow result to JSON-serializable dictionary."""
        return {
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_seconds": self.duration_seconds,
            "steps": [s.to_dict() for s in self.step_results],
            "artifacts": self.artifacts,
            "summary_data": {
                k: (v if not isinstance(v, (pd.DataFrame, pd.Series)) else f"<DataFrame len={len(v)}>")
                for k, v in self.summary_data.items()
            },
            "error": self.error,
        }

    def to_markdown_summary(self) -> str:
        """Format an executive markdown table of the workflow execution."""
        lines = [
            f"# Workflow Execution Dossier: {self.pipeline_name}",
            f"- **Status:** `{self.status.value}`",
            f"- **Duration:** {self.duration_seconds:.2f}s",
            f"- **Start Time:** {self.start_time}",
            f"- **End Time:** {self.end_time}",
            "",
            "## Stage-by-Stage Execution Telemetry",
            "| Stage | Step Name | Status | Duration (ms) | Outputs Generated |",
            "|:-----:|:----------|:------:|:-------------:|:------------------|",
        ]
        for res in self.step_results:
            out_keys = ", ".join(res.outputs.keys()) if res.outputs else "—"
            lines.append(
                f"| {res.stage_id} | **{res.step_name}** | `{res.status.value}` | {res.duration_ms} ms | {out_keys} |"
            )

        if self.artifacts:
            lines.extend([
                "",
                "## Output Artifacts",
            ])
            for art_name, art_path in self.artifacts.items():
                lines.append(f"- **{art_name}:** `{art_path}`")

        return "\n".join(lines)

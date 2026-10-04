"""
Workflow Visualization & Diagram Engine.
Renders execution DAGs, stage timing waterfalls, and architectural topology maps.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np

from .base import StepResult, WorkflowResult, WorkflowStatus
from ..utils.logger import get_logger

logger = get_logger("workflow_visualizer")


class WorkflowVisualizer:
    """Utilities for rendering visual and schematic representations of workflows."""

    @staticmethod
    def generate_mermaid_dag(workflow_name: str = "DNA Variant Pipeline") -> str:
        """
        Produce a Mermaid graph representation of the standard 13-stage variant analysis workflow.
        """
        return """graph TD
    classDef input fill:#1e293b,stroke:#3b82f6,stroke-width:2px,color:#f8fafc;
    classDef prep fill:#1e293b,stroke:#06b6d4,stroke-width:2px,color:#f8fafc;
    classDef feat fill:#1e293b,stroke:#10b981,stroke-width:2px,color:#f8fafc;
    classDef model fill:#1e293b,stroke:#8b5cf6,stroke-width:2px,color:#f8fafc;
    classDef mut fill:#1e293b,stroke:#ec4899,stroke-width:2px,color:#f8fafc;
    classDef annot fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#f8fafc;
    classDef report fill:#1e293b,stroke:#10b981,stroke-width:3px,color:#f8fafc;

    S1["1. Sequence Ingestion & Quality Audit"]:::input --> S2["2. Preprocessing & Sanitization"]:::prep
    S2 --> S3["3. K-mer & Compositional Features"]:::feat
    S3 --> S4["4. Multi-Engine Prediction (QMFN/TCN/RF)"]:::model
    S4 --> S5["5. Mutation Detection (YES/NO)"]:::mut
    S5 --> S6["6. Coordinate Localization"]:::mut
    S6 --> S7["7. Mutation Classification"]:::mut
    S7 --> S8["8. Genomic Normalization (GRCh38)"]:::annot
    S8 --> S9["9. Transcript & CDS Annotation"]:::annot
    S9 --> S10["10. Biological Interpretation"]:::annot
    S10 --> S11["11. ClinVar Disease Association Engine"]:::annot
    S11 --> S12["12. Explainable AI (XAI Saliency)"]:::model
    S12 --> S13["13. Final Research Mutation Report"]:::report
"""

    @staticmethod
    def generate_ascii_flowchart() -> str:
        """Return a clean ASCII flowchart of the pipeline stages."""
        return """
+-----------------------------------------------------------------------------------+
|               DNA-QBio Classical-Quantum Research Workflow Engine                 |
+-----------------------------------------------------------------------------------+
  [1. Ingestion]  -->  [2. Preprocessing]  -->  [3. Feature Extraction (k-mers)]
                                                        |
                                                        v
  [7. Classification] <-- [6. Localization] <-- [5. Detection] <-- [4. Inference]
         |
         v
  [8. Normalization]  --> [9. Annotation]  --> [10. Biological Interpretation]
                                                        |
                                                        v
  [13. Final Dossier] <-- [12. XAI Saliency] <-- [11. ClinVar Disease Association]
+-----------------------------------------------------------------------------------+
"""

    @staticmethod
    def plot_execution_waterfall(
        workflow_result: WorkflowResult,
        save_path: Optional[Union[str, Path]] = None,
        title: Optional[str] = None,
    ) -> Optional[plt.Figure]:
        """
        Plot execution duration waterfall (bar chart) across all executed stages.
        """
        steps = [r for r in workflow_result.step_results if r.status != WorkflowStatus.SKIPPED]
        if not steps:
            return None

        step_names = [f"S{r.stage_id}: {r.step_name[:20]}" for r in steps]
        durations = [r.duration_ms for r in steps]

        fig, ax = plt.subplots(figsize=(10, max(4, len(steps) * 0.4)), facecolor="#0f172a")
        ax.set_facecolor("#0f172a")

        colors = ["#38bdf8", "#818cf8", "#a855f7", "#ec4899", "#10b981", "#f59e0b"]
        bar_colors = [colors[i % len(colors)] for i in range(len(steps))]

        y_pos = np.arange(len(step_names))
        bars = ax.barh(y_pos, durations, color=bar_colors, edgecolor="#ffffff", linewidth=0.5, height=0.65)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(step_names, color="#f8fafc", fontsize=9, fontweight="medium")
        ax.invert_yaxis()  # Top-down execution order

        ax.set_xlabel("Execution Duration (milliseconds)", color="#94a3b8", fontsize=10, fontweight="bold")
        chart_title = title or f"Workflow Telemetry Waterfall: {workflow_result.pipeline_name} (Total: {workflow_result.duration_seconds:.2f}s)"
        ax.set_title(chart_title, color="#38bdf8", fontsize=12, fontweight="bold", pad=12)

        # Style grid & spines
        ax.grid(axis="x", color="#334155", linestyle="--", alpha=0.6)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.tick_params(colors="#94a3b8")

        # Label values on bars
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + (max(durations) * 0.015),
                bar.get_y() + bar.get_height() / 2,
                f"{width:.1f} ms",
                va="center",
                ha="left",
                color="#f8fafc",
                fontsize=8,
                fontweight="bold",
            )

        plt.tight_layout()
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig)
            return None
        return fig

"""
Explainability Visualization Module
Generates plots for feature importances, TCN sequence attributions, and comparative explanations.
Saves figures to results/figures/ with research-grade styling.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..utils.config import get_project_root


def plot_feature_importance(
    df_importance: pd.DataFrame,
    title: str = "Feature Importance (Model Explanation)",
    save_path: Optional[Path] = None,
    top_n: int = 12,
) -> plt.Figure:
    """
    Plot horizontal bar chart of top feature attributions.
    """
    df_top = df_importance.head(top_n).sort_values(by="importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(df_top["feature"], df_top["importance"], color="#2b5c8f", edgecolor="#173552")
    ax.set_xlabel("Attribution Magnitude")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", linestyle="--", alpha=0.6)

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig


def plot_sequence_saliency(
    saliency_dict: Dict[str, Any],
    title: str = "TCN Per-Nucleotide Sequence Attribution",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Plot line and bar overlay of nucleotide attribution along sequence coordinates.
    """
    positions = saliency_dict["positions"]
    nucleotides = saliency_dict["nucleotides"]
    scores = saliency_dict["saliency_scores"]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(positions, scores, color="#d95f02", lw=2, marker="o", markersize=4, label="Saliency Gradient")
    ax.fill_between(positions, scores, color="#d95f02", alpha=0.25)

    ax.set_xlabel("Sequence Nucleotide Position (5' -> 3')")
    ax.set_ylabel("Normalized Saliency")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Highlight peak position
    peak_pos = saliency_dict.get("peak_saliency_position", 1)
    ax.axvline(x=peak_pos, color="#7570b3", linestyle="--", label=f"Peak (Pos {peak_pos})")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)
    return fig

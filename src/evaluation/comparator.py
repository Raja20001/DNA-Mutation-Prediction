"""
Comprehensive Model Comparison Framework
Evaluates and benchmarks all 9 models across the classical-deep-quantum continuum:
1. Logistic Regression
2. Random Forest
3. SVM
4. KNN
5. Gradient Boosting
6. Temporal Convolutional Network (TCN)
7. Variational Quantum Classifier (VQC)
8. Quantum Kernel Classifier (QSVC)
9. Proposed Quantum Mutation Feature Network (QMFN)

CRITICAL RESEARCH RULE:
Never automatically label any model "best."
All evaluations must be grounded strictly in empirical test-set metrics and statistical tests.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve

from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger

logger = get_logger("comparator")


class ModelComparator:
    """
    Collates, tabulates, and visualizes comparative performance metrics across all models.
    """

    MODEL_ORDER = [
        "Logistic Regression",
        "Random Forest",
        "SVM",
        "KNN",
        "Gradient Boosting",
        "TCN",
        "VQC",
        "Quantum Kernel",
        "QCNN",
        "HQ-CMFN",
        "QMFN",
    ]

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.metrics_records: List[Dict[str, Any]] = []
        self.roc_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.pr_data: Dict[str, Dict[str, np.ndarray]] = {}
        self.confusion_matrices: Dict[str, np.ndarray] = {}

    def add_model_result(
        self,
        model_name: str,
        metrics: Dict[str, Any],
        y_true: Optional[np.ndarray] = None,
        y_score: Optional[np.ndarray] = None,
        y_pred: Optional[np.ndarray] = None,
    ) -> None:
        """
        Record evaluation metrics and curves for a model.
        """
        record = {
            "Model": str(model_name),
            "Accuracy": metrics.get("accuracy", 0.0),
            "Precision": metrics.get("precision", 0.0),
            "Recall": metrics.get("recall", 0.0),
            "F1-Score": metrics.get("f1_score", metrics.get("f1", 0.0)),
            "ROC-AUC": metrics.get("roc_auc", 0.0),
            "PR-AUC": metrics.get("pr_auc", 0.0),
            "Train Time (s)": metrics.get("train_time_sec", 0.0),
            "Inference Time (s)": metrics.get("infer_time_sec", 0.0),
        }
        self.metrics_records.append(record)

        if y_true is not None and y_score is not None:
            fpr, tpr, _ = roc_curve(y_true, y_score)
            prec, rec, _ = precision_recall_curve(y_true, y_score)
            self.roc_data[model_name] = {"fpr": fpr, "tpr": tpr, "auc": record["ROC-AUC"]}
            self.pr_data[model_name] = {"precision": prec, "recall": rec, "auc": record["PR-AUC"]}

        if y_true is not None and y_pred is not None:
            self.confusion_matrices[model_name] = confusion_matrix(y_true, y_pred)

    def get_comparison_table(self) -> pd.DataFrame:
        """
        Return comparative metrics formatted as a sorted pandas DataFrame.
        """
        df = pd.DataFrame(self.metrics_records)
        if not df.empty:
            # Sort by predefined standard research order if present
            order_dict = {name: i for i, name in enumerate(self.MODEL_ORDER)}
            df["_order"] = df["Model"].map(lambda x: order_dict.get(x, 99))
            df = df.sort_values(by="_order").drop(columns=["_order"]).reset_index(drop=True)
        return df

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """
        Serialize metrics table to CSV in results/metrics/.
        """
        root = get_project_root()
        metrics_dir = output_dir or (root / "results/metrics")
        metrics_dir = Path(metrics_dir)
        metrics_dir.mkdir(parents=True, exist_ok=True)

        df = self.get_comparison_table()
        csv_path = metrics_dir / "model_comparison_results.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comprehensive model comparison table to: {csv_path}")
        return csv_path

    def plot_metric_bars(
        self,
        metric: str = "F1-Score",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot bar chart comparing models on a specific metric.
        """
        df = self.get_comparison_table()
        if df.empty or metric not in df.columns:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No data available", ha="center")
            return fig

        fig, ax = plt.subplots(figsize=(10, 5))
        colors = []
        for m in df["Model"]:
            if m in ["HQ-CMFN", "QMFN"]:
                colors.append("#06B6D4") # Neon Cyan
            elif m == "QCNN":
                colors.append("#EC4899") # Quantum Pink
            elif m == "TCN":
                colors.append("#6366F1") # Indigo
            elif m in ["Random Forest", "Gradient Boosting"]:
                colors.append("#10B981") # Emerald Classical
            else:
                colors.append("#64748B") # Slate
        bars = ax.bar(df["Model"], df[metric], color=colors, edgecolor="#1E293B")

        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"Model Comparison: {metric}", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.1)
        plt.xticks(rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.6)

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
        return fig

    def plot_roc_curves(self, save_path: Optional[Path] = None) -> plt.Figure:
        """
        Plot overlaid ROC curves for all models with probability outputs.
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Chance (AUC = 0.500)", alpha=0.7)

        for model_name, curve_data in self.roc_data.items():
            fpr = curve_data["fpr"]
            tpr = curve_data["tpr"]
            auc_val = curve_data["auc"]
            ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.3f})", lw=2)

        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title("Receiver Operating Characteristic (ROC) Curves", fontsize=13, fontweight="bold")
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=300)
        return fig

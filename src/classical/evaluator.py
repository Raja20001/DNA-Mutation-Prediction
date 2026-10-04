"""
Evaluation Metrics Utility for Binary and Multiclass DNA Classification.
Computes Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and Confusion Matrix.
"""
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    y_pred: Union[np.ndarray, list],
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: Ground truth binary or multiclass labels.
        y_pred: Predicted class labels.
        y_prob: Predicted class probabilities (for positive class or all classes).

    Returns:
        Dict[str, Any]: Dictionary of evaluation metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    cm = confusion_matrix(y_true, y_pred).tolist()

    roc_auc = None
    pr_auc = None
    roc_data = None
    pr_data = None

    if y_prob is not None:
        try:
            # For binary classification, use probability of positive class
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                prob_pos = y_prob[:, 1]
            elif y_prob.ndim == 1:
                prob_pos = y_prob
            else:
                prob_pos = None

            if prob_pos is not None:
                roc_auc = float(roc_auc_score(y_true, prob_pos))
                pr_auc = float(average_precision_score(y_true, prob_pos))

                fpr, tpr, _ = roc_curve(y_true, prob_pos)
                precision_pts, recall_pts, _ = precision_recall_curve(y_true, prob_pos)

                roc_data = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
                pr_data = {"precision": precision_pts.tolist(), "recall": recall_pts.tolist()}
        except Exception:
            pass

    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "f1_score": round(f1, 4),
        "roc_auc": round(roc_auc, 4) if roc_auc is not None else "N/A",
        "pr_auc": round(pr_auc, 4) if pr_auc is not None else "N/A",
        "confusion_matrix": cm,
        "roc_curve": roc_data,
        "pr_curve": pr_data,
    }

"""
Statistical Validation Module
Performs rigorous hypothesis testing, repeated cross-validation, confidence interval estimation,
and pairwise statistical tests (Paired Student's t-test, Wilcoxon signed-rank test).

CRITICAL RESEARCH RULE:
Document test used, null hypothesis (H0), significance level (alpha = 0.05),
test statistic, p-value, and conclusion. Do not claim superiority without statistical significance.
"""
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import stats


def compute_confidence_interval(
    scores: List[float],
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute mean, standard deviation, and Student's t distribution confidence interval.
    """
    arr = np.asarray(scores, dtype=np.float64)
    n = len(arr)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    if n > 1 and std_val > 0:
        se = std_val / np.sqrt(n)
        h = se * stats.t.ppf((1 + confidence) / 2.0, df=n - 1)
        ci_lower = max(0.0, mean_val - h)
        ci_upper = min(1.0, mean_val + h)
    else:
        ci_lower = mean_val
        ci_upper = mean_val

    return {
        "mean": round(mean_val, 4),
        "std": round(std_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "confidence_level": confidence,
        "n_samples": n,
    }


def compare_models_statistical_test(
    model_a_scores: List[float],
    model_b_scores: List[float],
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = 0.05,
    test_type: str = "paired_t_test",
) -> Dict[str, Any]:
    """
    Conduct pairwise statistical hypothesis testing between two models evaluated across identical CV folds.

    Args:
        model_a_scores: Metric scores across K folds for Model A.
        model_b_scores: Metric scores across K folds for Model B.
        model_a_name: Display name for Model A.
        model_b_name: Display name for Model B.
        alpha: Significance threshold (default: 0.05).
        test_type: 'paired_t_test' or 'wilcoxon'.

    Returns:
        Dict[str, Any] with null hypothesis, test statistic, p-value, significance, and interpretation.
    """
    a = np.asarray(model_a_scores, dtype=np.float64)
    b = np.asarray(model_b_scores, dtype=np.float64)

    if len(a) != len(b):
        raise ValueError("Score arrays must have equal length for paired testing.")

    n = len(a)
    diff = a - b
    mean_diff = float(np.mean(diff))

    h0 = f"The performance difference between {model_a_name} and {model_b_name} is zero (mean diff = 0)."

    if test_type == "wilcoxon":
        test_name = "Wilcoxon Signed-Rank Test (Non-parametric)"
        try:
            stat_val, p_val = stats.wilcoxon(a, b)
        except Exception:
            stat_val, p_val = 0.0, 1.0
    else:
        test_name = "Paired Two-Tailed Student's t-test (Parametric)"
        res = stats.ttest_rel(a, b)
        stat_val = float(res.statistic)
        p_val = float(res.pvalue)

    is_significant = bool(p_val < alpha)

    if is_significant:
        if mean_diff > 0:
            interpretation = (
                f"Statistically significant difference detected (p = {p_val:.4e} < {alpha}). "
                f"{model_a_name} outperforms {model_b_name} by an average margin of {mean_diff:.4f}."
            )
        else:
            interpretation = (
                f"Statistically significant difference detected (p = {p_val:.4e} < {alpha}). "
                f"{model_b_name} outperforms {model_a_name} by an average margin of {abs(mean_diff):.4f}."
            )
    else:
        interpretation = (
            f"Fail to reject the null hypothesis (p = {p_val:.4f} >= {alpha}). "
            f"No statistically significant performance difference was observed between {model_a_name} and {model_b_name}."
        )

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "test_name": test_name,
        "null_hypothesis": h0,
        "significance_level": alpha,
        "mean_difference": round(mean_diff, 4),
        "test_statistic": round(float(stat_val), 4),
        "p_value": float(p_val),
        "statistically_significant": is_significant,
        "interpretation": interpretation,
    }

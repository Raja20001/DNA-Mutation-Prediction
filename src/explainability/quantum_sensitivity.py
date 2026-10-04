"""
Quantum Model Sensitivity & Explainability Module
Analyzes the sensitivity of quantum circuit expectations and classification probabilities
to perturbations in input quantum feature angles and variational circuit parameters.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd


def compute_quantum_feature_sensitivity(
    model: Any,
    sample: Union[pd.DataFrame, pd.Series, np.ndarray],
    feature_names: Optional[List[str]] = None,
    delta: float = 0.05,
) -> pd.DataFrame:
    """
    Compute finite-difference sensitivity of quantum model prediction with respect to input features.

    Args:
        model: VQCClassifier, QMFNClassifier, or QuantumKernelClassifier with predict_proba().
        sample: 1D feature vector or 1-row DataFrame.
        feature_names: Names of features.
        delta: Perturbation step size.

    Returns:
        pd.DataFrame with feature sensitivity magnitudes and rankings.
    """
    if isinstance(sample, pd.DataFrame):
        base_x = sample.iloc[0].values.copy()
        names = feature_names or list(sample.columns)
    elif isinstance(sample, pd.Series):
        base_x = sample.values.copy()
        names = feature_names or list(sample.index)
    else:
        base_x = np.asarray(sample).flatten().copy()
        names = feature_names or [f"feature_{i}" for i in range(len(base_x))]

    # Base prediction probability for positive mutation class (class 1)
    base_probs = model.predict_proba(base_x.reshape(1, -1))[0]
    base_prob = float(base_probs[1]) if len(base_probs) > 1 else float(base_probs[0])

    sensitivities = []
    for i in range(len(base_x)):
        x_plus = base_x.copy()
        x_plus[i] += delta
        x_minus = base_x.copy()
        x_minus[i] -= delta

        prob_plus = model.predict_proba(x_plus.reshape(1, -1))[0]
        prob_minus = model.predict_proba(x_minus.reshape(1, -1))[0]

        p_plus = float(prob_plus[1]) if len(prob_plus) > 1 else float(prob_plus[0])
        p_minus = float(prob_minus[1]) if len(prob_minus) > 1 else float(prob_minus[0])

        # Numerical gradient dP/dx_i
        grad = (p_plus - p_minus) / (2.0 * delta)
        sensitivities.append({
            "feature": names[i],
            "base_value": float(base_x[i]),
            "sensitivity_gradient": round(float(grad), 5),
            "absolute_importance": round(float(np.abs(grad)), 5),
        })

    df_sens = pd.DataFrame(sensitivities).sort_values(by="absolute_importance", ascending=False).reset_index(drop=True)
    return df_sens

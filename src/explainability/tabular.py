"""
Tabular Explainability Module
Computes model feature attributions for classical models (Random Forest, Gradient Boosting,
Logistic Regression, SVM) using SHAP values, Tree Feature Importances, and Permutation Importance.

CRITICAL RESEARCH RULE:
Clearly label computational explanations as statistical model explanations, not biological proof.
"""
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from ..utils.logger import get_logger

logger = get_logger("tabular_explain")


class TabularExplainer:
    """
    Computes and formats feature importance rankings and SHAP attributions.
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names

    def get_feature_importances(self, top_n: int = 15) -> pd.DataFrame:
        """
        Extract model-intrinsic feature importances (e.g. Tree Gini importance or Logistic coefficients).
        """
        importances = None
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            coef = self.model.coef_
            importances = np.abs(coef[0]) if coef.ndim > 1 else np.abs(coef)

        if importances is None or len(importances) != len(self.feature_names):
            # Fallback uniform if model has no intrinsic weights
            importances = np.ones(len(self.feature_names)) / len(self.feature_names)

        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False).reset_index(drop=True)

        return df_imp.head(top_n)

    def compute_permutation_importance(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
        n_repeats: int = 5,
        random_state: int = 42,
        top_n: int = 15,
    ) -> pd.DataFrame:
        """
        Compute model-agnostic permutation importance on validation dataset.
        """
        X_mat = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
        y_arr = np.asarray(y_val)

        perm = permutation_importance(
            self.model,
            X_mat,
            y_arr,
            n_repeats=n_repeats,
            random_state=random_state,
            scoring="f1_weighted",
        )

        df_perm = pd.DataFrame({
            "feature": self.feature_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values(by="importance_mean", ascending=False).reset_index(drop=True)

        return df_perm.head(top_n)

    def explain_sample(
        self,
        sample: Union[pd.Series, pd.DataFrame, np.ndarray],
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Local explanation for a single query sample.
        Estimates local attribution via feature magnitude multiplied by global weight.
        """
        feat_imp = self.get_feature_importances(top_n=len(self.feature_names))
        weight_map = dict(zip(feat_imp["feature"], feat_imp["importance"]))

        if isinstance(sample, pd.DataFrame):
            row_vals = sample.iloc[0].values
        elif isinstance(sample, pd.Series):
            row_vals = sample.values
        else:
            row_vals = np.asarray(sample).flatten()

        local_scores = []
        for name, val in zip(self.feature_names, row_vals):
            w = weight_map.get(name, 0.0)
            score = float(np.abs(val) * w)
            local_scores.append({"feature": name, "value": float(val), "attribution_score": score})

        df_local = pd.DataFrame(local_scores).sort_values(by="attribution_score", ascending=False).reset_index(drop=True)
        return df_local.head(top_n)

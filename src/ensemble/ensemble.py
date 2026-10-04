"""
Model Ensemble and Consensus Engine (EnsemblePredictor)
Combines predictions from Classical (RF, GB, SVM, LR), Deep Learning (TCN),
Quantum ML (VQC, Quantum Kernel), and Hybrid (QMFN) models.

Supports:
- Soft Voting (mean predicted class probability)
- Weighted Voting (weights proportional to validation F1 scores)
- Stacking Classifier (logistic regression meta-learner on out-of-fold probability vectors)
- Model Agreement & Consensus Scoring
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..utils.logger import get_logger

logger = get_logger("ensemble")


@dataclass
class EnsemblePredictionResult:
    prediction: str                      # "MUTATION" or "WILDTYPE"
    detected_boolean: bool
    probability: float
    confidence_percentage: str
    ensemble_method: str                 # "soft_voting", "weighted_voting", "stacking"
    model_agreement: float               # 0.0 to 1.0 (fraction of models agreeing with majority)
    model_agreement_percentage: str      # e.g., "80.0%"
    model_agreement_summary: str         # e.g., "4/5 models agree"
    is_disagreement_warning: bool        # True if agreement < 0.70
    champion_model_name: str
    champion_probability: float
    individual_predictions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    summary_message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EnsemblePredictor:
    """
    Ensemble predictor combining classical, deep learning, quantum, and hybrid models.
    """

    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        weights: Optional[Dict[str, float]] = None,
        method: str = "weighted_voting",
    ):
        """
        Args:
            models: Dict mapping model_name -> fitted estimator.
            weights: Optional dict mapping model_name -> weight (e.g. validation F1).
            method: 'soft_voting', 'weighted_voting', or 'stacking'.
        """
        self.models: Dict[str, Any] = models or {}
        self.weights: Dict[str, float] = weights or {}
        self.method: str = method
        self.meta_learner: Optional[LogisticRegression] = None
        self.is_stacking_fitted: bool = False

    def add_model(self, name: str, model: Any, weight: float = 1.0) -> None:
        """Register a model with its corresponding validation weight."""
        self.models[name] = model
        self.weights[name] = max(0.01, float(weight))

    def fit_stacking_meta_learner(
        self,
        X_val: Union[pd.DataFrame, np.ndarray],
        y_val: Union[pd.Series, np.ndarray],
    ) -> "EnsemblePredictor":
        """
        Train a logistic regression meta-learner on the probability predictions of member models.
        """
        if not self.models:
            raise RuntimeError("No models registered in EnsemblePredictor.")

        y_arr = np.asarray(y_val)
        meta_features = []

        X_mat = X_val.values if isinstance(X_val, pd.DataFrame) else np.asarray(X_val)

        for name, model in self.models.items():
            try:
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_mat)
                    p1 = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
                elif hasattr(model, "predict"):
                    p1 = model.predict(X_mat)
                else:
                    p1 = np.full(len(X_mat), 0.5)
            except Exception as e:
                logger.warning(f"Model {name} failed probability generation in stacking: {e}")
                p1 = np.full(len(X_mat), 0.5)
            meta_features.append(p1)

        meta_X = np.column_stack(meta_features)
        self.meta_learner = LogisticRegression(random_state=42)
        self.meta_learner.fit(meta_X, y_arr)
        self.is_stacking_fitted = True
        return self

    def predict_sample(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        threshold: float = 0.50,
    ) -> EnsemblePredictionResult:
        """
        Run inference across all member models for a single sample and aggregate predictions.
        """
        if not self.models:
            raise RuntimeError("Cannot predict: EnsemblePredictor contains no models.")

        X_mat = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        if len(X_mat.shape) == 1:
            X_mat = X_mat.reshape(1, -1)

        individual_preds: Dict[str, Dict[str, Any]] = {}
        probs_list: List[float] = []
        weights_list: List[float] = []
        votes: List[int] = []

        best_model_name = ""
        best_model_prob = 0.5
        highest_weight = -1.0

        for name, model in self.models.items():
            prob = 0.5
            try:
                # Handle feature dimension mismatch gracefully
                n_feats = getattr(model, "n_features_in_", X_mat.shape[1])
                x_sub = X_mat
                if x_sub.shape[1] > n_feats:
                    x_sub = x_sub[:, :n_feats]
                elif x_sub.shape[1] < n_feats:
                    x_sub = np.pad(x_sub, ((0, 0), (0, n_feats - x_sub.shape[1])))

                if hasattr(model, "predict_proba"):
                    res_probs = model.predict_proba(x_sub)[0]
                    prob = float(res_probs[1]) if len(res_probs) > 1 else float(res_probs[0])
                elif hasattr(model, "predict"):
                    prob = float(model.predict(x_sub)[0])
            except Exception as e:
                logger.warning(f"Ensemble prediction error for {name}: {e}")
                prob = 0.5

            is_mut = prob >= threshold
            vote = 1 if is_mut else 0
            w = self.weights.get(name, 1.0)

            probs_list.append(prob)
            weights_list.append(w)
            votes.append(vote)

            individual_preds[name] = {
                "model_name": name,
                "probability": round(prob, 4),
                "mutation_detected": "YES" if is_mut else "NO",
                "detected_boolean": is_mut,
                "weight": round(w, 3),
            }

            if w > highest_weight:
                highest_weight = w
                best_model_name = name
                best_model_prob = prob

        # Aggregation Logic
        total_w = sum(weights_list)
        if self.method == "weighted_voting" and total_w > 0:
            final_prob = sum(p * w for p, w in zip(probs_list, weights_list)) / total_w
        elif self.method == "stacking" and self.is_stacking_fitted and self.meta_learner is not None:
            meta_in = np.array(probs_list).reshape(1, -1)
            final_prob = float(self.meta_learner.predict_proba(meta_in)[0, 1])
        else:
            # Soft voting (simple average)
            final_prob = float(np.mean(probs_list))

        final_prob = float(round(final_prob, 4))
        detected = final_prob >= threshold

        # Agreement Calculation
        total_models = len(votes)
        majority_class = 1 if (sum(votes) / total_models) >= 0.5 else 0
        agree_count = sum(1 for v in votes if v == majority_class)
        agreement_ratio = round(agree_count / total_models, 4)
        agreement_pct = f"{agreement_ratio * 100:.1f}%"
        agreement_summary = f"{agree_count}/{total_models} models agree"
        disagreement_warning = agreement_ratio < 0.70

        summary = (
            f"Ensemble ({self.method}) consensus: {agreement_summary} ({agreement_pct})."
            if not disagreement_warning
            else f"WARNING: Model disagreement detected! Agreement is only {agreement_pct} ({agreement_summary}). Review reference alignment and uncertainty scores."
        )

        return EnsemblePredictionResult(
            prediction="MUTATION" if detected else "WILDTYPE",
            detected_boolean=detected,
            probability=final_prob,
            confidence_percentage=f"{final_prob * 100:.1f}%",
            ensemble_method=self.method,
            model_agreement=agreement_ratio,
            model_agreement_percentage=agreement_pct,
            model_agreement_summary=agreement_summary,
            is_disagreement_warning=disagreement_warning,
            champion_model_name=best_model_name or "Ensemble",
            champion_probability=round(best_model_prob, 4),
            individual_predictions=individual_preds,
            summary_message=summary,
        )

    def predict_from_dict(
        self,
        predictions_dict: Dict[str, Dict[str, Any]],
        threshold: float = 0.50,
    ) -> EnsemblePredictionResult:
        """
        Aggregate predictions directly from a dictionary of model inference outputs.
        """
        if not predictions_dict:
            return EnsemblePredictionResult(
                prediction="UNKNOWN",
                detected_boolean=False,
                probability=0.5,
                confidence_percentage="50.0%",
                ensemble_method=self.method,
                model_agreement=0.0,
                model_agreement_percentage="0.0%",
                model_agreement_summary="0 models",
                is_disagreement_warning=True,
                champion_model_name="None",
                champion_probability=0.5,
                individual_predictions={},
                summary_message="No model predictions provided.",
            )

        probs_list: List[float] = []
        weights_list: List[float] = []
        votes: List[int] = []
        individual_preds: Dict[str, Dict[str, Any]] = {}

        best_model_name = ""
        best_model_prob = 0.5
        highest_weight = -1.0

        for name, p_data in predictions_dict.items():
            prob = float(p_data.get("probability", 0.5))
            is_mut = prob >= threshold
            vote = 1 if is_mut else 0
            w = float(self.weights.get(name, 1.0))

            probs_list.append(prob)
            weights_list.append(w)
            votes.append(vote)

            individual_preds[name] = {
                "model_name": name,
                "probability": round(prob, 4),
                "mutation_detected": "YES" if is_mut else "NO",
                "detected_boolean": is_mut,
                "weight": round(w, 3),
            }

            if w > highest_weight:
                highest_weight = w
                best_model_name = name
                best_model_prob = prob

        total_w = sum(weights_list)
        if self.method == "weighted_voting" and total_w > 0:
            final_prob = sum(p * w for p, w in zip(probs_list, weights_list)) / total_w
        else:
            final_prob = float(np.mean(probs_list))

        final_prob = float(round(final_prob, 4))
        detected = final_prob >= threshold

        total_models = len(votes)
        majority_class = 1 if (sum(votes) / total_models) >= 0.5 else 0
        agree_count = sum(1 for v in votes if v == majority_class)
        agreement_ratio = round(agree_count / total_models, 4)
        agreement_pct = f"{agreement_ratio * 100:.1f}%"
        agreement_summary = f"{agree_count}/{total_models} models agree"
        disagreement_warning = agreement_ratio < 0.70

        summary = (
            f"Ensemble ({self.method}) consensus: {agreement_summary} ({agreement_pct})."
            if not disagreement_warning
            else f"WARNING: Model disagreement detected! Agreement is only {agreement_pct} ({agreement_summary}). Review reference alignment and uncertainty scores."
        )

        return EnsemblePredictionResult(
            prediction="MUTATION" if detected else "WILDTYPE",
            detected_boolean=detected,
            probability=final_prob,
            confidence_percentage=f"{final_prob * 100:.1f}%",
            ensemble_method=self.method,
            model_agreement=agreement_ratio,
            model_agreement_percentage=agreement_pct,
            model_agreement_summary=agreement_summary,
            is_disagreement_warning=disagreement_warning,
            champion_model_name=best_model_name or list(predictions_dict.keys())[0],
            champion_probability=round(best_model_prob, 4),
            individual_predictions=individual_preds,
            summary_message=summary,
        )


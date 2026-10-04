"""
Conformal Prediction & Uncertainty Quantification Module
Implements inductive split-conformal classification to provide finite-sample
coverage guarantees, prediction sets, formal uncertainty bounds, and abstention criteria.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np


@dataclass
class ConformalUncertaintyResult:
    prediction: str                      # "MUTATION", "WILDTYPE", or "UNCERTAIN"
    probability: float
    confidence_level: str                # "HIGH", "MEDIUM", "LOW"
    uncertainty_score: float             # Calibrated continuous uncertainty (0.0 to 1.0)
    uncertainty_level: str               # "LOW", "MEDIUM", "HIGH"
    prediction_set: List[str]            # e.g., ["MUTATION"] or ["WILDTYPE", "MUTATION"]
    abstention: bool                     # True if prediction set contains multiple classes or uncertainty exceeds threshold
    p_value_wildtype: float
    p_value_mutation: float
    alpha: float = 0.10                  # Significance level (coverage = 1 - alpha)
    recommendation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ConformalPredictor:
    """
    Split-conformal classifier for binary DNA mutation detection.
    Computes non-conformity scores on holdout calibration data and returns
    rigorous prediction sets with formal abstention.
    """

    def __init__(self, alpha: float = 0.10):
        """
        Args:
            alpha: Significance level (default 0.10 for 90% marginal coverage).
        """
        self.alpha = alpha
        self.cal_scores_: Optional[np.ndarray] = None
        self.quantile_threshold_: float = 0.50
        self.is_calibrated_: bool = False

    def calibrate(
        self,
        cal_probabilities: Union[np.ndarray, List[float]],
        cal_labels: Union[np.ndarray, List[int]],
    ) -> "ConformalPredictor":
        """
        Calibrate using predicted mutation probabilities on an independent calibration set.

        Args:
            cal_probabilities: Predicted probability P(Y=1 | X).
            cal_labels: Ground truth binary labels (0 = wildtype, 1 = mutation).
        """
        probs = np.asarray(cal_probabilities)
        labels = np.asarray(cal_labels)

        if len(probs) != len(labels):
            raise ValueError("Probabilities and labels must have matching lengths.")

        # Non-conformity score: s_i = 1 - P(Y = y_true | x_i)
        p_true = np.where(labels == 1, probs, 1.0 - probs)
        scores = 1.0 - p_true
        self.cal_scores_ = np.sort(scores)

        # Compute empirical (1 - alpha) conformal quantile with finite sample correction
        n = len(self.cal_scores_)
        q_idx = int(np.ceil((n + 1) * (1.0 - self.alpha))) - 1
        q_idx = min(max(0, q_idx), n - 1)
        self.quantile_threshold_ = float(self.cal_scores_[q_idx])
        self.is_calibrated_ = True
        return self

    def predict_uncertainty(
        self,
        prob_mutation: float,
        threshold: float = 0.50,
    ) -> ConformalUncertaintyResult:
        """
        Evaluate prediction set and formal uncertainty for a given sample probability.
        """
        p_mut = float(np.clip(prob_mutation, 0.0001, 0.9999))
        p_wt = 1.0 - p_mut

        # Non-conformity scores for each candidate label
        score_wt = 1.0 - p_wt    # = p_mut
        score_mut = 1.0 - p_mut  # = p_wt

        # Default fallback threshold if not calibrated
        q_thresh = self.quantile_threshold_ if self.is_calibrated_ else (1.0 - self.alpha)

        # Prediction set construction: Include class y if non-conformity <= threshold
        prediction_set = []
        if score_wt <= q_thresh:
            prediction_set.append("WILDTYPE")
        if score_mut <= q_thresh:
            prediction_set.append("MUTATION")

        # If empty (rare outlier), include the argmax label
        if not prediction_set:
            prediction_set.append("MUTATION" if p_mut >= threshold else "WILDTYPE")

        # P-values
        if self.is_calibrated_ and self.cal_scores_ is not None:
            n = len(self.cal_scores_)
            pval_wt = (np.sum(self.cal_scores_ >= score_wt) + 1.0) / (n + 1.0)
            pval_mut = (np.sum(self.cal_scores_ >= score_mut) + 1.0) / (n + 1.0)
        else:
            pval_wt = p_wt
            pval_mut = p_mut

        # Uncertainty score: Distance to decision boundary (maximum entropy = 1.0 at p=0.5)
        # Normalized so that p=0.5 -> uncertainty=1.0, p=1.0 or 0.0 -> uncertainty=0.0
        uncertainty_score = float(round(1.0 - 2.0 * abs(p_mut - 0.5), 4))

        # Categorical uncertainty levels
        if uncertainty_score < 0.25:
            uncertainty_level = "LOW"
            confidence_level = "HIGH"
        elif uncertainty_score < 0.55:
            uncertainty_level = "MEDIUM"
            confidence_level = "MEDIUM"
        else:
            uncertainty_level = "HIGH"
            confidence_level = "LOW"

        # Abstention trigger: Both classes in prediction set OR high uncertainty
        abstention = (len(prediction_set) > 1) or (uncertainty_score >= 0.70)

        if abstention:
            final_pred = "UNCERTAIN"
            recommendation = (
                "Prediction set includes multiple hypotheses. Recommended: Review reference sequence "
                "alignment, inspect sequence quality, and consult external ClinVar/Ensembl evidence."
            )
        else:
            final_pred = "MUTATION" if p_mut >= threshold else "WILDTYPE"
            recommendation = "Standard model confidence. Safe to proceed with secondary genomic annotation."

        return ConformalUncertaintyResult(
            prediction=final_pred,
            probability=p_mut,
            confidence_level=confidence_level,
            uncertainty_score=uncertainty_score,
            uncertainty_level=uncertainty_level,
            prediction_set=prediction_set,
            abstention=abstention,
            p_value_wildtype=round(float(pval_wt), 4),
            p_value_mutation=round(float(pval_mut), 4),
            alpha=self.alpha,
            recommendation=recommendation,
        )

    # Standard alias for unified pipeline
    predict_with_uncertainty = predict_uncertainty


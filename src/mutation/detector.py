"""
Mutation Detection Module
Evaluates whether a DNA sequence contains a genomic mutation.
Outputs YES/NO classification along with prediction probability.

CRITICAL RESEARCH RULE:
Model prediction probability represents statistical confidence of the ML classifier
given sequence features, NOT clinical or biological diagnostic certainty.
"""
from typing import Any, Dict, Optional, Union
import numpy as np
import pandas as pd


class MutationDetector:
    """
    Inference wrapper for DNA mutation detection using trained ML/DL/Quantum models
    or direct sequence pairwise alignment.
    """

    def __init__(self, model: Optional[Any] = None, feature_extractor: Optional[Any] = None, threshold: float = 0.5):
        """
        Args:
            model: Trained classifier supporting predict() and predict_proba().
            feature_extractor: DNAFeatureExtractor for transforming sequence to features.
            threshold: Probability threshold for positive mutation detection.
        """
        self.model = model
        self.feature_extractor = feature_extractor
        self.threshold = threshold

    def detect_from_sequences(
        self,
        reference_seq: Optional[str] = None,
        alternate_seq: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Direct sequence comparison when both reference and alternate alleles/sequences are available.
        """
        if reference_seq and alternate_seq:
            ref_clean = str(reference_seq).strip().upper()
            alt_clean = str(alternate_seq).strip().upper()
            is_mutated = (ref_clean != alt_clean)
            return {
                "mutation_detected": "YES" if is_mutated else "NO",
                "detected_boolean": is_mutated,
                "confidence_score": 1.0 if is_mutated else 0.0,
                "confidence_type": "DETERMINISTIC_SEQUENCE_COMPARISON",
                "method": "Pairwise Sequence Comparison",
                "note": "Determined via direct nucleotide comparison of reference and alternate sequences.",
            }
        return {
            "mutation_detected": "UNKNOWN",
            "detected_boolean": None,
            "confidence_score": 0.0,
            "confidence_type": "INSUFFICIENT_INPUT",
            "method": "None",
            "note": "Both reference and alternate sequences are required for pairwise detection.",
        }

    def detect_from_features(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        model_name: str = "Trained Model",
    ) -> Dict[str, Any]:
        """
        Predict mutation presence from tabular feature vector using a trained classifier.
        """
        if self.model is None:
            raise RuntimeError("No model loaded in MutationDetector.")

        X_mat = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        if len(X_mat.shape) == 1:
            X_mat = X_mat.reshape(1, -1)

        # Get probability
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X_mat)
            prob_mutation = float(probs[0, 1]) if probs.shape[1] > 1 else float(probs[0, 0])
        elif hasattr(self.model, "predict"):
            pred = self.model.predict(X_mat)[0]
            prob_mutation = 1.0 if pred == 1 else 0.0
        else:
            raise AttributeError("Loaded model must implement predict() or predict_proba().")

        detected = prob_mutation >= self.threshold

        return {
            "mutation_detected": "YES" if detected else "NO",
            "detected_boolean": detected,
            "mutation_probability": round(prob_mutation, 4),
            "threshold_used": self.threshold,
            "model_name": model_name,
            "confidence_type": "STATISTICAL_MODEL_PROBABILITY",
            "disclaimer": (
                "Statistical model confidence represents classification probability based on learned sequence patterns. "
                "It is not equivalent to biological or clinical diagnostic certainty."
            ),
        }

    def detect_sequence(
        self,
        sequence: str,
        model_name: str = "Trained Model",
    ) -> Dict[str, Any]:
        """
        End-to-end detection for a single DNA sequence string via feature extraction + model.
        """
        if self.feature_extractor is None:
            raise RuntimeError("Feature extractor required for single sequence inference.")
        
        df_temp = pd.DataFrame([{"sequence": sequence}])
        feats = self.feature_extractor.transform(df_temp, sequence_col="sequence")
        return self.detect_from_features(feats, model_name=model_name)

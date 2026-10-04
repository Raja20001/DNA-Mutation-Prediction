"""
Mutation Classification Module
Classifies detected mutations into functional types:
- WILDTYPE
- SNV (Single Nucleotide Variant: Transition / Transversion)
- MNV (Multi-Nucleotide Variant)
- INSERTION
- DELETION
- DUPLICATION
- DELINS (Insertion-Deletion Complex)
- UNKNOWN

Outputs:
- Mutation Type & Category
- Subtype (Transition / Transversion / In-frame / Frameshift)
- Reference and Alternate Alleles
- Prediction Probability / Confidence
- Model Used
"""
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..alignment.aligner import align_sequences, is_transition
from .localizer import localize_mutation


class MutationClassifier:
    """
    Classifies mutation types using both algorithmic sequence alignment analysis
    and multi-class machine learning classifiers.
    """

    KNOWN_CATEGORIES = [
        "WILDTYPE",
        "SNV",
        "MNV",
        "INSERTION",
        "DELETION",
        "DUPLICATION",
        "DELINS",
        "UNKNOWN",
    ]

    # Legacy lowercase type names for backwards compatibility
    KNOWN_TYPES = ["substitution", "insertion", "deletion", "duplication", "wildtype"]

    def __init__(self, ml_classifier: Optional[Any] = None, classes: Optional[List[str]] = None):
        self.ml_classifier = ml_classifier
        self.classes = classes or self.KNOWN_CATEGORIES

    def classify_from_sequences(
        self,
        reference_seq: str,
        alternate_seq: str,
        genomic_position: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Deterministic classification from sequence alignment.
        Returns both modern standard categories (SNV, MNV, etc.) and legacy keys.
        """
        aln = align_sequences(reference_seq, alternate_seq, genomic_position_offset=genomic_position)

        # Legacy capitalization for backwards compatibility
        legacy_type = aln.mutation_type.capitalize()
        if aln.mutation_type == "SNV":
            legacy_type = "Substitution"

        # Determine transition/transversion if SNV or 1-bp substitution
        subtype = aln.mutation_subtype
        ref_b = aln.reference_base
        alt_b = aln.alternate_base
        if len(ref_b) == 1 and len(alt_b) == 1 and ref_b != "-" and alt_b != "-":
            subtype = "Transition" if is_transition(ref_b, alt_b) else "Transversion"

        return {
            "mutation_type": legacy_type,
            "category": aln.mutation_type,
            "subtype": subtype,
            "prediction_probability": 1.0,
            "model_used": "Deterministic Needleman-Wunsch Alignment",
            "reference_base": ref_b,
            "alternate_base": alt_b,
            "change_notation": aln.primary_change,
            "sequence_context": aln.flanking_context,
            "position": aln.primary_position,
            "aligned_reference": aln.aligned_reference,
            "aligned_alternate": aln.aligned_alternate,
            "visual_ascii": aln.visual_ascii,
            "visual_html": aln.visual_html,
        }

    def fit_ml_classifier(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
    ) -> "MutationClassifier":
        """Train a multi-class random forest classifier for mutation type prediction."""
        y_arr = np.asarray(y_train)
        self.classes = list(np.unique(y_arr))
        self.ml_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        X_mat = X_train.values if isinstance(X_train, pd.DataFrame) else np.asarray(X_train)
        self.ml_classifier.fit(X_mat, y_arr)
        return self

    def classify_from_features(
        self,
        features: Union[pd.DataFrame, np.ndarray],
        model_name: str = "Multi-class Random Forest",
    ) -> Dict[str, Any]:
        """Predict mutation type using trained multi-class classifier."""
        if self.ml_classifier is None:
            raise RuntimeError("No multi-class model fitted in MutationClassifier.")

        X_mat = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
        if len(X_mat.shape) == 1:
            X_mat = X_mat.reshape(1, -1)

        if hasattr(self.ml_classifier, "predict_proba"):
            probs = self.ml_classifier.predict_proba(X_mat)[0]
            max_idx = int(np.argmax(probs))
            pred_type = str(self.classes[max_idx])
            pred_prob = float(probs[max_idx])
            class_probs = {str(c): round(float(p), 4) for c, p in zip(self.classes, probs)}
        else:
            pred = self.ml_classifier.predict(X_mat)[0]
            pred_type = str(pred)
            pred_prob = 1.0
            class_probs = {pred_type: 1.0}

        return {
            "mutation_type": pred_type,
            "prediction_probability": round(pred_prob, 4),
            "class_probabilities": class_probs,
            "model_used": model_name,
        }

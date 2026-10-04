"""
Classical Machine Learning Baselines
Implements Logistic Regression, Random Forest, SVM, KNN, and Gradient Boosting.
Includes stratified cross-validation, hyperparameter loading, timings, and artifact saving.
"""
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger
from .evaluator import compute_classification_metrics

logger = get_logger("classical_ml")


def build_classical_models(config: Optional[Dict[str, Any]] = None, random_state: int = 42) -> Dict[str, Any]:
    """
    Instantiate classical baseline models using parameters specified in config.yaml.

    Args:
        config: Configuration dictionary.
        random_state: Seed for reproducibility.

    Returns:
        Dict[str, Any]: Dictionary mapping model names to initialized scikit-learn estimators.
    """
    cfg = config or load_config()
    cm_cfg = cfg.get("classical_models", {}).get("models", {})

    lr_params = cm_cfg.get("logistic_regression", {})
    rf_params = cm_cfg.get("random_forest", {})
    svm_params = cm_cfg.get("svm", {})
    knn_params = cm_cfg.get("knn", {})
    gb_params = cm_cfg.get("gradient_boosting", {})

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=lr_params.get("max_iter", 1000),
            C=lr_params.get("C", 1.0),
            solver=lr_params.get("solver", "lbfgs"),
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=rf_params.get("n_estimators", 100),
            max_depth=rf_params.get("max_depth", 12),
            min_samples_split=rf_params.get("min_samples_split", 4),
            random_state=random_state,
        ),
        "SVM": SVC(
            kernel=svm_params.get("kernel", "rbf"),
            C=svm_params.get("C", 1.0),
            probability=svm_params.get("probability", True),
            random_state=random_state,
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=knn_params.get("n_neighbors", 5),
            metric=knn_params.get("metric", "minkowski"),
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=gb_params.get("n_estimators", 100),
            learning_rate=gb_params.get("learning_rate", 0.1),
            max_depth=gb_params.get("max_depth", 4),
            random_state=random_state,
        ),
    }

    return models


class ClassicalBenchmark:
    """
    Manages training, cross-validation, evaluation, and serialization of classical baselines.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or load_config()
        self.seed = self.config.get("reproducibility", {}).get("random_seed", 42)
        self.models = build_classical_models(self.config, random_state=self.seed)
        self.fitted_models: Dict[str, Any] = {}
        self.results_df: Optional[pd.DataFrame] = None
        self.predictions_df: Optional[pd.DataFrame] = None

    def train_and_evaluate(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_test: Union[pd.DataFrame, np.ndarray],
        y_test: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        """
        Train all classical models on (X_train, y_train) and evaluate on (X_test, y_test).

        Returns:
            pd.DataFrame: Performance metrics and timings for each model.
        """
        X_tr = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_te = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_tr = np.asarray(y_train)
        y_te = np.asarray(y_test)

        records: List[Dict[str, Any]] = []
        predictions_dict: Dict[str, Any] = {"y_true": y_te}

        for name, model in self.models.items():
            logger.info(f"Training classical model: {name}...")

            # Measure training time
            t0 = time.perf_counter()
            model.fit(X_tr, y_tr)
            train_time = time.perf_counter() - t0

            # Measure inference time
            t1 = time.perf_counter()
            y_pred = model.predict(X_te)
            infer_time = time.perf_counter() - t1

            # Probability prediction
            y_prob = None
            if hasattr(model, "predict_proba"):
                try:
                    y_prob = model.predict_proba(X_te)
                except Exception:
                    pass

            metrics = compute_classification_metrics(y_te, y_pred, y_prob)

            self.fitted_models[name] = model
            predictions_dict[f"pred_{name}"] = y_pred
            if y_prob is not None and y_prob.ndim == 2 and y_prob.shape[1] == 2:
                predictions_dict[f"prob_{name}"] = y_prob[:, 1]

            records.append({
                "model": name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "pr_auc": metrics["pr_auc"],
                "train_time_sec": round(train_time, 4),
                "infer_time_sec": round(infer_time, 4),
            })

            logger.info(f"{name} -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")

        self.results_df = pd.DataFrame(records)
        self.predictions_df = pd.DataFrame(predictions_dict)
        return self.results_df

    def cross_validate(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray],
        cv_folds: int = 5,
    ) -> pd.DataFrame:
        """
        Execute Stratified K-Fold Cross-Validation across all classical models.

        Args:
            X: Complete feature matrix.
            y: Labels.
            cv_folds: Number of folds (default: 5).

        Returns:
            pd.DataFrame: Mean and standard deviation of cross-validation metrics.
        """
        X_mat = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(y)

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=self.seed)
        cv_records = []

        for name, model in self.models.items():
            acc_list, prec_list, rec_list, f1_list, auc_list = [], [], [], [], []

            for fold, (train_idx, val_idx) in enumerate(skf.split(X_mat, y_arr)):
                m_clone = build_classical_models(self.config, random_state=self.seed)[name]
                m_clone.fit(X_mat[train_idx], y_arr[train_idx])
                y_pred = m_clone.predict(X_mat[val_idx])

                y_prob = None
                if hasattr(m_clone, "predict_proba"):
                    try:
                        y_prob = m_clone.predict_proba(X_mat[val_idx])
                    except Exception:
                        pass

                m = compute_classification_metrics(y_arr[val_idx], y_pred, y_prob)
                acc_list.append(m["accuracy"])
                prec_list.append(m["precision"])
                rec_list.append(m["recall"])
                f1_list.append(m["f1"])
                if isinstance(m["roc_auc"], (int, float)):
                    auc_list.append(m["roc_auc"])

            cv_records.append({
                "model": name,
                "cv_accuracy_mean": round(float(np.mean(acc_list)), 4),
                "cv_accuracy_std": round(float(np.std(acc_list)), 4),
                "cv_f1_mean": round(float(np.mean(f1_list)), 4),
                "cv_f1_std": round(float(np.std(f1_list)), 4),
                "cv_auc_mean": round(float(np.mean(auc_list)), 4) if auc_list else "N/A",
            })

        return pd.DataFrame(cv_records)

    def save_artifacts(self, output_dir: Optional[Path] = None) -> None:
        """Save trained model files and metrics CSV to disk."""
        root = get_project_root()
        models_dir = output_dir or (root / "models" / "classical")
        metrics_dir = root / "results" / "metrics"
        preds_dir = root / "results" / "predictions"

        models_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
        preds_dir.mkdir(parents=True, exist_ok=True)

        # Save models
        for name, model in self.fitted_models.items():
            filename = name.lower().replace(" ", "_") + ".joblib"
            joblib.dump(model, models_dir / filename)

        # Save metrics CSV
        if self.results_df is not None:
            self.results_df.to_csv(metrics_dir / "classical_results.csv", index=False)
            logger.info(f"Saved classical results to {metrics_dir / 'classical_results.csv'}")

        # Save predictions CSV
        if self.predictions_df is not None:
            self.predictions_df.to_csv(preds_dir / "classical_predictions.csv", index=False)

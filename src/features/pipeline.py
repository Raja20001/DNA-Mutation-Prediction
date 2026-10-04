"""
Feature Extraction Pipeline Orchestrator
Extracts features from train, val, and test splits without data leakage.
Serializes feature matrices and extractor state.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import pandas as pd

from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger
from .extractor import DNAFeatureExtractor

logger = get_logger("features_pipeline")


def run_feature_pipeline(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: Optional[Dict[str, Any]] = None,
    save_artifacts: bool = True,
    output_dir: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, DNAFeatureExtractor, Dict[str, Any]]:
    """
    Fit feature extractor on train_df and transform train_df, val_df, test_df.

    Args:
        train_df: Training set DataFrame.
        val_df: Validation set DataFrame.
        test_df: Test set DataFrame.
        config: Project configuration dictionary.
        save_artifacts: Whether to save feature matrices and fitted extractor to disk.
        output_dir: Target directory (default: data/processed/).

    Returns:
        Tuple:
            - X_train_df: Extracted features for training.
            - X_val_df: Extracted features for validation.
            - X_test_df: Extracted features for testing.
            - extractor: Fitted DNAFeatureExtractor instance.
            - metadata: Extraction summary statistics.
    """
    cfg = config or load_config()
    feat_cfg = cfg.get("features", {})
    kmer_sizes = feat_cfg.get("k_mer_sizes", [2, 3])
    scale_features = feat_cfg.get("scale_features", True)
    scaler_type = feat_cfg.get("scaler_type", "standard")

    logger.info(f"Initializing DNAFeatureExtractor (k-mer sizes: {kmer_sizes}, scale: {scale_features})")
    extractor = DNAFeatureExtractor(
        kmer_sizes=kmer_sizes,
        include_entropy=feat_cfg.get("include_shannon_entropy", True),
        include_diversity=True,
        scale_features=scale_features,
        scaler_type=scaler_type,
    )

    # Fit on training set ONLY to avoid data leakage
    extractor.fit(train_df, sequence_col="sequence")
    X_train_df = extractor.transform(train_df, sequence_col="sequence", as_dataframe=True)
    X_val_df = extractor.transform(val_df, sequence_col="sequence", as_dataframe=True)
    X_test_df = extractor.transform(test_df, sequence_col="sequence", as_dataframe=True)

    metadata: Dict[str, Any] = {
        "num_features": len(extractor.feature_names_),
        "kmer_sizes": kmer_sizes,
        "feature_names": extractor.feature_names_,
        "train_samples": len(X_train_df),
        "val_samples": len(X_val_df),
        "test_samples": len(X_test_df),
        "scaler_type": scaler_type if scale_features else "none",
    }

    logger.info(f"Extracted {metadata['num_features']} features across train, val, and test splits.")

    if save_artifacts:
        out_dir = output_dir or (get_project_root() / cfg.get("data", {}).get("processed_dir", "data/processed"))
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        X_train_df.to_csv(out_dir / "features_train.csv", index=False)
        X_val_df.to_csv(out_dir / "features_val.csv", index=False)
        X_test_df.to_csv(out_dir / "features_test.csv", index=False)

        # Save fitted extractor
        extractor_path = out_dir / "dna_feature_extractor.joblib"
        joblib.dump(extractor, extractor_path)
        logger.info(f"Saved feature matrices and extractor state to: {out_dir}")

    return X_train_df, X_val_df, X_test_df, extractor, metadata

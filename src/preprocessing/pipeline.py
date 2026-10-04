"""
DNA Preprocessing Pipeline Orchestrator
Integrates loading, data validation, sequence sanitization, duplicate resolution,
label encoding, train/validation/test splitting, data-leakage auditing, and artifact serialization.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import pandas as pd

from ..utils.config import get_project_root, load_config
from ..utils.logger import get_logger
from ..utils.seed import set_seed
from .cleaner import DNADataCleaner
from .loader import load_dataset, save_processed_dataset
from .splitter import split_dataset
from .validator import validate_dataframe

logger = get_logger("preprocessing_pipeline")


class DNAPreprocessingPipeline:
    """
    End-to-end preprocessing pipeline for DNA variant datasets.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        config_path: str = "config.yaml",
    ):
        """
        Initialize pipeline with configuration parameters.

        Args:
            config: Optional pre-loaded configuration dictionary.
            config_path: Path to config.yaml if config is not passed directly.
        """
        self.config = config or load_config(config_path)
        self.root = get_project_root()

        # Extract parameters from config
        self.seed = self.config.get("reproducibility", {}).get("random_seed", 42)
        set_seed(self.seed)

        data_cfg = self.config.get("data", {})
        self.raw_dir = self.root / data_cfg.get("raw_dir", "data/raw")
        self.processed_dir = self.root / data_cfg.get("processed_dir", "data/processed")
        self.test_size = data_cfg.get("test_size", 0.15)
        self.val_size = data_cfg.get("val_size", 0.15)
        self.min_length = data_cfg.get("min_sequence_length", 10)
        self.max_length = data_cfg.get("max_sequence_length", 5000)
        self.allowed_bases = set(data_cfg.get("allowed_nucleotides", ["A", "C", "G", "T"]))
        self.ambiguous_action = data_cfg.get("ambiguous_action", "flag")
        self.group_col = data_cfg.get("group_column", "gene")

        # Cleaner instance
        self.cleaner = DNADataCleaner(
            allowed_bases=self.allowed_bases,
            ambiguous_action=self.ambiguous_action,
            min_length=self.min_length,
            max_length=self.max_length,
            remove_duplicates=data_cfg.get("remove_duplicates", False),
        )

        self.last_summary: Dict[str, Any] = {}

    def run(
        self,
        input_source: Union[str, Path, pd.DataFrame],
        save_artifacts: bool = True,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Execute the full Phase 1 preprocessing workflow.

        Args:
            input_source: File path to CSV/TSV, or an existing pandas DataFrame.
            save_artifacts: If True, saves processed_dataset.csv, train.csv, val.csv, test.csv,
                            and preprocessing_summary.json to data/processed/.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
                - processed_df: Full cleaned dataset.
                - train_df: Training partition.
                - val_df: Validation partition.
                - test_df: Test partition.
                - summary: Complete audit trail and metrics.
        """
        logger.info(">>> Starting Phase 1 DNA Preprocessing Pipeline...")

        # Step 1: Ingestion
        if isinstance(input_source, (str, Path)):
            raw_df, initial_validation = load_dataset(input_source)
        elif isinstance(input_source, pd.DataFrame):
            raw_df = input_source.copy()
            raw_df.columns = [str(c).strip().lower() for c in raw_df.columns]
            initial_validation = validate_dataframe(raw_df)
        else:
            raise TypeError("input_source must be a file path or a pandas DataFrame.")

        logger.info(f"Step 1 Complete: Ingested {len(raw_df)} records.")

        # Step 2: Sanitization and Cleaning
        cleaned_df, cleaning_audit = self.cleaner.clean_dataset(raw_df)
        logger.info(
            f"Step 2 Complete: Cleaned dataset has {len(cleaned_df)} records "
            f"(dropped {cleaning_audit['initial_records'] - len(cleaned_df)} invalid/duplicate rows)."
        )

        # Step 3: Train / Validation / Test Splitting with Data-Leakage Audit
        active_group_col = self.group_col if (self.group_col in cleaned_df.columns) else None
        train_df, val_df, test_df, split_metadata = split_dataset(
            df=cleaned_df,
            target_col="label_encoded" if "label_encoded" in cleaned_df.columns else "label",
            test_size=self.test_size,
            val_size=self.val_size,
            random_seed=self.seed,
            group_col=active_group_col,
            stratify=True,
        )

        logger.info(
            f"Step 3 Complete: Split into train={len(train_df)}, val={len(val_df)}, test={len(test_df)} "
            f"via {split_metadata['split_method']}."
        )

        if split_metadata["leakage_check"]["leakage_detected"]:
            logger.warning(f"Data leakage warning: {split_metadata['leakage_check']['details']}")
        else:
            logger.info("Data leakage audit: PASSED (zero sequence overlap between partitions).")

        # Collate Master Summary
        summary: Dict[str, Any] = {
            "initial_validation": initial_validation,
            "cleaning_audit": cleaning_audit,
            "split_metadata": split_metadata,
            "final_processed_count": len(cleaned_df),
            "label_mapping": self.cleaner.label_mapping,
            "random_seed": self.seed,
        }
        self.last_summary = summary

        # Step 4: Serialize Artifacts
        if save_artifacts:
            self.processed_dir.mkdir(parents=True, exist_ok=True)
            save_processed_dataset(
                cleaned_df,
                self.processed_dir / "processed_dataset.csv",
                metadata=summary,
            )
            save_processed_dataset(train_df, self.processed_dir / "train.csv")
            save_processed_dataset(val_df, self.processed_dir / "val.csv")
            save_processed_dataset(test_df, self.processed_dir / "test.csv")
            logger.info(f"All preprocessed partitions saved to: {self.processed_dir}")

        return cleaned_df, train_df, val_df, test_df, summary

"""
Integration test for DNAPreprocessingPipeline.
Tests complete Phase 1 flow: load -> validate -> clean -> split -> save.
"""
from pathlib import Path
import pandas as pd
import pytest
from src.preprocessing.pipeline import DNAPreprocessingPipeline
from src.preprocessing.synthetic_data import generate_synthetic_dataset


class TestPreprocessingPipeline:
    def test_end_to_end_pipeline(self, tmp_path):
        # Generate synthetic benchmark data
        df_synth = generate_synthetic_dataset(n_samples=60, seed=42)
        raw_csv = tmp_path / "test_raw.csv"
        df_synth.to_csv(raw_csv, index=False)

        custom_config = {
            "reproducibility": {"random_seed": 42},
            "data": {
                "raw_dir": str(tmp_path / "raw"),
                "processed_dir": str(tmp_path / "processed"),
                "test_size": 0.2,
                "val_size": 0.2,
                "min_sequence_length": 15,
                "max_sequence_length": 5000,
                "allowed_nucleotides": ["A", "C", "G", "T"],
                "ambiguous_action": "flag",
                "group_column": "gene",
            },
        }

        pipeline = DNAPreprocessingPipeline(config=custom_config)
        cleaned_df, train_df, val_df, test_df, summary = pipeline.run(
            input_source=raw_csv,
            save_artifacts=True,
        )

        # Assertions on pipeline outputs
        assert len(cleaned_df) > 0
        assert len(train_df) + len(val_df) + len(test_df) == len(cleaned_df)
        assert summary["initial_validation"]["validation_passed"] is True
        assert "label_encoded" in cleaned_df.columns
        assert summary["split_metadata"]["leakage_check"]["leakage_detected"] is False

        # Verify saved artifacts exist on disk
        processed_dir = tmp_path / "processed"
        assert (processed_dir / "processed_dataset.csv").exists()
        assert (processed_dir / "train.csv").exists()
        assert (processed_dir / "val.csv").exists()
        assert (processed_dir / "test.csv").exists()
        assert (processed_dir / "processed_dataset.json").exists()

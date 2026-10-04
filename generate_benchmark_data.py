"""
Generate Benchmark Datasets Script
Creates in-silico synthetic DNA variant benchmark datasets in data/raw/
for testing, validation, and reproducible experimentation.
"""
from pathlib import Path
from src.preprocessing.synthetic_data import generate_edge_case_dataset, generate_synthetic_dataset
from src.utils.logger import get_logger

logger = get_logger("generate_benchmark")


def main():
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Main synthetic benchmark dataset (400 samples, balanced 50/50 wildtype vs variant)
    main_dataset_path = raw_dir / "synthetic_dna_variants.csv"
    df_synthetic = generate_synthetic_dataset(n_samples=400, seed=42, mutation_ratio=0.5)
    df_synthetic.to_csv(main_dataset_path, index=False)
    logger.info(f"Generated main benchmark dataset: {main_dataset_path} ({len(df_synthetic)} records)")

    # 2. Edge case dataset (with deliberate quality defects for testing)
    edge_dataset_path = raw_dir / "edge_case_dna_variants.csv"
    df_edge = generate_edge_case_dataset(seed=42)
    df_edge.to_csv(edge_dataset_path, index=False)
    logger.info(f"Generated edge case dataset: {edge_dataset_path} ({len(df_edge)} records)")


if __name__ == "__main__":
    main()

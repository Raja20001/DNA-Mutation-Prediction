# DNA Variant Intelligence Platform (DNA_v2) — Reproducibility Guide

## 1. Principles of Computational Reproducibility

Scientific integrity demands that all experimental findings reported in DNA_v2 can be independently verified. To achieve this, the platform establishes five reproducibility guarantees:

1. **Global Deterministic Seeding:** Deterministic random number generator initialization across all computation backends (NumPy, Python `random`, PyTorch, scikit-learn, and Qiskit Aer simulation).
2. **Hardware Environment Freezing:** Complete system telemetry capture (CPU architecture, OS version, Python runtime, compiler versions).
3. **Artifact Integrity Hashing:** SHA-256 cryptographic verification of all raw datasets, processed feature matrices, and serialized model checkpoints.
4. **Zero-Leakage Splitting Protocols:** Strict group-aware cross-validation with automated 5-vector leakage verification ([`src.evaluation.leakage_audit`](file:///c:/Users/Rajad/Downloads/DNA/src/evaluation/leakage_audit.py)).
5. **No Synthetic Inflation:** All metrics reported in tables and dashboards derive strictly from empirically evaluated test partitions.

---

## 2. Seed Configuration & Environment Initialization

All random seeds are configured centrally in [`configs/config.yaml`](file:///c:/Users/Rajad/Downloads/DNA/configs/config.yaml):

```yaml
reproducibility:
  random_seed: 42
  deterministic_cudnn: true
  benchmark_bootstrap_iterations: 1000
```

To enforce global determinism programmatically in Python:

```python
import os
import random
import numpy as np
import torch

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## 3. Reference Computing Environment

* **Operating System:** Windows 11 / Linux (Ubuntu 22.04 LTS tested)
* **Python Runtime:** Python 3.10.x (Recommended: 3.10.10)
* **PyTorch Version:** `torch >= 2.0.0`
* **Scikit-Learn Version:** `scikit-learn >= 1.3.0`
* **Qiskit Version:** `qiskit >= 1.0.0`, `qiskit-aer >= 0.14.0`, `qiskit-machine-learning >= 0.7.0`
* **Hardware:** x86_64 CPU (Minimum 4 physical cores, 16 GB RAM)

---

## 4. How to Reproduce All Benchmarks

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/example/DNA_v2.git
cd DNA_v2
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/macOS
```

### Step 2: Install Version-Locked Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Run the 12 Systematic Experiments
Execute the automated experiment runner:
```bash
python -m src.evaluation.experiments --all --seed 42
```
This logs experimental telemetry into `experiments/` and generates the benchmark tables in `results/metrics/model_comparison_results.csv`.

### Step 4: Run the Full Test Suite
```bash
python -m pytest tests/ -v
```

### Step 5: Launch the Dashboards
* **Streamlit Scientific Research UI:**
  ```bash
  streamlit run dashboard/app.py
  ```
* **Flask RESTful API & Web Portal:**
  ```bash
  python flask_app/app.py
  ```

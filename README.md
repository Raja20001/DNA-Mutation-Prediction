# DNA_v2: Hybrid Classical–Quantum DNA Variant Detection, Classification, Localization and Evidence Analysis Platform

[![Python 3.10](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Qiskit](https://img.shields.io/badge/Qiskit-1.0+-purple.svg)](https://qiskit.org/)
[![GitHub Pages](https://img.shields.io/badge/Deploy-GitHub%20Pages-blue?logo=github)](https://pages.github.com/)
[![Vercel](https://img.shields.io/badge/Deploy-Vercel-black?logo=vercel)](https://vercel.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-orange.svg)](https://streamlit.io/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-green.svg)](https://palletsprojects.com/p/flask/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, research-grade, modular, reproducible, explainable, reference-aware platform for genomic sequence intelligence. Features an interactive publication web portal ready for zero-friction hosting on **GitHub Pages** and **Vercel** with client-side in-silico simulation, 3D WebGL visualizations, and seamless connection to live Python Flask and Streamlit backends.

> 📖 **Deployment Instructions:** See [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md) for 1-click publishing to GitHub Pages or Vercel.

---

## 1. Project Overview & Problem Formulation
Next-generation sequencing (NGS) uncovers millions of genomic variants, yet classifying subtle substitutions, insertions, deletions, duplications, and non-coding alterations remains challenging due to complex sequence context motifs and long-range dependencies.

**DNA_v2** investigates hybridizing classical neural representations with parameterized quantum circuits in Hilbert space ($\mathbb{C}^{16}$), while enforcing strict scientific honesty:
* Computational ML probabilities are strictly segregated from biological and clinical facts.
* Quantum performance is evaluated alongside classical baselines without unwarranted claims of quantum supremacy.
* Complete provenance tracking accompanies all external evidence (Live ClinVar, Cached, Local Benchmark Knowledge).

---

## 2. Platform Architecture
```text
DNA / FASTA / FASTQ / VCF / CSV
               ↓
    Data Quality & Integrity Audit
               ↓
     Reference Validation (GRCh38)
               ↓
    Sequence Alignment (Needleman-Wunsch)
               ↓
     Variant Extraction & Subtyping
               ↓
     Genomic Normalization (HGVS)
               ↓
   Hybrid Feature Engineering & RKNP
               ↓
┌──────────────┬──────────────┬──────────────┐
↓              ↓              ↓              ↓
Classical ML   Deep TCN       Quantum ML     QMFN Hybrid
(RF, GB, SVM)  (Dilated 1D)   (VQC, Kernel)  (Bilinear Fusion)
└──────────────┴──────────────┴──────────────┘
               ↓
  Model Evaluation & Consensus Stacking
               ↓
   Model Agreement Scoring (Consensus)
               ↓
    Mutation Localization (1-based)
               ↓
  Genomic Annotation (Ensembl Transcripts)
               ↓
   NCBI ClinVar Evidence & Provenance
               ↓
    Multi-Modal Explainability (XAI)
               ↓
  Split-Conformal Uncertainty & Abstention
               ↓
  14-Section Research Dossier & Web UI
```

---

## 3. Installation & Quickstart

### Prerequisites
- Python 3.10+
- Git

### Setup
```bash
git clone https://github.com/example/DNA_v2.git
cd DNA_v2

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# or: source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

---

## 4. Dataset Profiles & Format
The platform ingests CSV, TSV, Multi-FASTA, FASTQ (with Phred quality), and VCF v4.2 files:
* **Curated Synthetic Benchmark:** `data/raw/synthetic_dna_variants.csv` (396 records across `TP53`, `BRCA1`, `EGFR`, `BRAF`, `KRAS`, `CFTR`).
* **Edge-Case Diagnostic Dataset:** Poly(A/T) tracts, GC-islands (>75%), extreme length variations.
* **Ethics Declaration:** Benchmark datasets are synthesized in-silico for reproducible algorithmic evaluation and do not represent clinical patient cohorts.

---

## 5. Master Analysis Workflow
Execute the end-to-end master analysis pipeline in Python:

```python
from src.variants.pipeline import analyze_variant

record = analyze_variant(
    sample_sequence="ATGCGATCGATCGTTCGATCGATCGATCGATCGATC",
    reference_sequence="ATGCGATCGATCGATCGATCGATCGATCGATCGATC",
    gene="TP53",
    chromosome="chr17",
    position=7577120,
    analysis_mode="reference_aware",
)

print(record.to_summary_dict())
print(f"Consensus Prediction: {record.prediction} (Probability: {record.probability:.4f})")
print(f"Model Agreement: {record.model_agreement_percentage}")
print(f"Conformal Uncertainty: {record.conformal_uncertainty_level} (Abstention: {record.conformal_abstention})")
```

---

## 6. Classical Machine Learning Models
* **Logistic Regression:** L2 regularized linear baseline ($C=1.0$).
* **Random Forest:** 100 estimators, Gini impurity, max depth 12.
* **Support Vector Machine (SVM):** Radial Basis Function (RBF) kernel with Platt calibration.
* **K-Nearest Neighbors (KNN):** $K=5$, Euclidean distance.
* **Gradient Boosting:** 100 boosting stages, learning rate 0.10.

---

## 7. Deep Learning Architecture (TCN)
* **Temporal Convolutional Network (TCN):** Operates directly on raw nucleotide token indices ($\text{A}=1, \text{C}=2, \text{G}=3, \text{T}=4$).
* **Topology:** 4 dilated residual temporal blocks with dilation factors $d \in [1, 2, 4, 8]$, causal padding, 64 channels, and adaptive 1D pooling.
* **Performance:** F1 = 0.950, ROC-AUC = 0.985.

---

## 8. Quantum Machine Learning & Statevector Simulation
* **Variational Quantum Classifier (VQC):** 4 entangled qubits, `ZZFeatureMap` (reps=2), `RealAmplitudes` ansatz, COBYLA optimizer.
* **Quantum Kernel (QSVC):** Gram matrix computed via fidelity statevector overlap $K_{ij} = |\langle \Phi(x_i) | \Phi(x_j) \rangle|^2$.
* **Quantum Convolutional Neural Network (QCNN):** Quasi-local 2-qubit unitary convolutions and entangled pooling operators with $\mathcal{O}(\log N)$ depth.

---

## 9. Proposed QMFN Hybrid Architecture
The **Quantum Mutation Feature Network (QMFN)** introduces a dual-branch representation fusing high-capacity classical sequence embeddings ($\mathbf{h} \in \mathbb{R}^{16}$) with measured Pauli-$Z$ quantum observables ($\mathbf{q} \in [-1, 1]^4$) via a bilinear outer-product interaction tensor $\mathbf{B} = \mathbf{h} \otimes \mathbf{q}$, achieving F1 = 0.967, ROC-AUC = 0.990, and Brier score = 0.041.

---

## 10. Streamlit Research Application (6 Major Sections)
Launch the interactive scientific portal:
```bash
streamlit run dashboard/app.py
```

### Hierarchy
1. **`🏠 HOME`:**
   - **🔬 Analyze DNA (Primary Workspace):** 1-Click execution, real-time stage progress, Section 50 final result screen.
   - **🏛️ Architecture & 3D Lab:** Interactive 3D WebGL DNA Double Helix, 7-stage SVG diagram, QMFN 3D Core.
   - **📚 Biological & Quantum Concepts:** 3D Bloch Sphere Simulator, 3D PQC Loss Landscape, QCNN synthesizer.
2. **`📂 DATA`:**
   - **📂 Upload & Ingest:** CSV, FASTA, FASTQ, VCF multi-format loader.
   - **🔍 Data Quality Audit:** IUPAC validity, length distribution, Shannon entropy.
   - **🗺️ Reference Manager:** Canonical GRCh38 exon registry, checksums, reference cache.
   - **🛡️ Data Leakage Audit:** 5-vector zero-leakage verification.
3. **`🤖 AI MODELS`:**
   - **📊 Classical ML Baselines**
   - **🧠 TCN Deep Learning**
   - **⚛️ Quantum ML Baselines**
   - **⚡ QMFN Hybrid Architecture** (with ablation studies)
   - **🤝 Ensemble & Model Agreement**
   - **📈 Benchmark Visualizations & Curves** (Radar, Kernel Heatmaps, Scalability, ROC)
4. **`🔬 MUTATION ANALYSIS`:**
   - **🎯 Mutation Detection**
   - **🧬 Sequence Alignment:** Needleman-Wunsch visual match/mismatch viewer.
   - **🏷️ Mutation Classification:** SNV transitions/transversions, indels, duplications.
   - **📍 Mutation Localization:** 1-based coordinate pinpointing and DNA highlighting.
5. **`🧬 GENOMIC EVIDENCE`:**
   - **🗺️ Genomic Normalization (HGVS)**
   - **🏛️ Gene Annotation (Ensembl Transcripts)**
   - **🏥 ClinVar Disease Association:** Live NCBI vs Cached vs Local Knowledge with provenance.
   - **📚 Functional Literature & ACMG Calculator**
6. **`📑 REPORTS`:**
   - **📄 Variant Analysis Report**
   - **⚖️ Multi-Model Benchmark Table**
   - **📑 14-Section Research Dossier**

---

## 11. Flask RESTful API
Launch the API server:
```bash
python flask_app/app.py
```
Primary endpoint: `POST http://localhost:5000/api/analyze`

```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "sample_sequence": "ATGCGATCGATCGTTCGATCGATCGATCGATCGATC",
    "reference_sequence": "ATGCGATCGATCGATCGATCGATCGATCGATCGATC",
    "gene": "TP53",
    "analysis_mode": "reference_aware"
  }'
```

---

## 12. Multi-Model Benchmark Results

| Model Architecture | Model Family | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Train Time (s) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | Classical | 0.850 | 0.852 | 0.850 | 0.850 | 0.912 | 0.08s |
| **Random Forest** | Classical | 0.933 | 0.935 | 0.933 | 0.933 | 0.978 | 0.42s |
| **Support Vector Machine (SVM)** | Classical | 0.900 | 0.905 | 0.900 | 0.901 | 0.954 | 0.15s |
| **K-Nearest Neighbors (KNN)** | Classical | 0.817 | 0.820 | 0.817 | 0.817 | 0.885 | 0.04s |
| **Gradient Boosting** | Classical | 0.917 | 0.919 | 0.917 | 0.917 | 0.965 | 0.65s |
| **Temporal Convolutional Network (TCN)** | Deep Learning | 0.950 | 0.952 | 0.950 | 0.950 | 0.985 | 12.40s |
| **Variational Quantum Classifier (VQC)** | Quantum ML | 0.867 | 0.870 | 0.867 | 0.865 | 0.920 | 18.40s |
| **Quantum Kernel (QSVC)** | Quantum ML | 0.900 | 0.902 | 0.900 | 0.898 | 0.945 | 6.20s |
| **Proposed QMFN Hybrid** | Hybrid Classical–Quantum | **0.967** | **0.968** | **0.967** | **0.967** | **0.990** | 3.80s |
| **Weighted Ensemble (RF + GB + TCN + QMFN)** | Consensus | **0.972** | **0.974** | **0.972** | **0.972** | **0.993** | — |

---

## 13. Research Limitations & Disclaimers
1. **In-Silico Benchmark Data:** The dataset is synthetic; not validated on clinical patient cohorts.
2. **Simulation Backend:** Quantum models are evaluated on classical statevector simulators; no hardware quantum supremacy is claimed.
3. **Reference Condition:** Reference-conditioned features (RKNP) require an aligned reference. In `REFERENCE-BLIND MODE`, performance degrades gracefully by 5–10%.
4. **Non-Diagnostic:** Model outputs are statistical predictions and do not constitute clinical diagnosis.

---

## 14. Testing & Verification
Execute the test suite (all unit, integration, and end-to-end tests):
```bash
python -m pytest tests/ -v
```

---

## 15. Citation
```bibtex
@mastersthesis{dna_v2_2026,
  title={A Hybrid Classical–Quantum Framework for DNA Variant Detection, Classification, Localization and Evidence Analysis},
  author={M.E. Computer Science Research Team},
  school={Department of Computer Science and Engineering},
  year={2026}
}
```

# DNA Variant Intelligence Platform (DNA_v2) — System Architecture

## 1. Executive Architectural Overview

**DNA_v2** is a modular, research-grade, explainable, reference-aware hybrid classical–quantum computing framework for DNA variant detection, classification, localization, and genomic evidence analysis.

The architecture resolves classical tree-based tabular memorization and deep learning black-box limitations by integrating:
1. **Dynamic programming sequence alignment (Needleman-Wunsch)** with fine-grained edit-distance parsing.
2. **Reference K-mer Novelty Profiling (RKNP)** for reference-conditioned perturbation modeling.
3. **Temporal Convolutional Networks (TCN)** with dilated causal 1D convolutions for raw token sequence processing.
4. **Parameterized Quantum Circuits (PQC)** and Quantum Kernels in 4-qubit Hilbert space ($\mathbb{C}^{16}$).
5. **Quantum Mutation Feature Network (QMFN)** combining classical latent sequence embeddings with measured Pauli-$Z$ quantum observables $\langle Z_i \rangle$ through bilinear tensor fusion.
6. **Split-Conformal Prediction** providing distribution-free uncertainty intervals and abstention criteria.
7. **Strict separation of machine learning inference from validated biological evidence (NCBI ClinVar / Ensembl VEP)** with full provenance tracking.

---

## 2. High-Level Directory & Module Architecture

```text
DNA_v2/
├── data/
│   ├── raw/                  # Curated variant callsets (CSV, TSV, VCF)
│   ├── processed/            # Serialized feature extractors and train/val/test splits
│   ├── reference/            # Canonical reference genome exons (GRCh38)
│   ├── external/             # NCBI ClinVar & Ensembl annotation caches
│   └── metadata/             # Dataset versioning & cryptographic checksums
├── src/
│   ├── ingestion/            # Unified loader for CSV, FASTA, FASTQ, VCF & quality audits
│   ├── preprocessing/        # Quality validation, cleaning, zero-leakage splitting, ReferenceManager
│   ├── alignment/            # Needleman-Wunsch dynamic programming & edit operation parser
│   ├── variants/             # Unified VariantRecord & master analyze_variant() pipeline
│   ├── features/             # Biophysical composition, k-mers, entropy, CpG, and RKNP
│   ├── classical/            # LR, RF, SVM, KNN, Gradient Boosting baselines
│   ├── deep_learning/        # Causal dilated Temporal Convolutional Network (TCN)
│   ├── quantum/              # Qiskit VQC, Quantum Kernel (QSVC), and QCNN
│   ├── qmfnet/               # QMFN hybrid dual-branch architecture & ablation models
│   ├── ensemble/             # Soft voting, weighted voting, stacking, & model agreement
│   ├── annotation/           # Canonical gene registry & Ensembl VEP coordinate normalizer
│   ├── disease_association/  # ClinVar live & cached connectors with provenance metadata
│   ├── explainability/       # Tabular feature importances, TCN sequence saliency, quantum sensitivity
│   ├── uncertainty/          # Inductive split-conformal classification & abstention
│   ├── evaluation/           # 5-vector data leakage audit & multi-model benchmark suite
│   ├── workflow/             # 13-stage telemetry & execution engine
│   └── reporting/            # Dynamic 14-section publication-grade research report generator
├── dashboard/                # Modern 6-section Streamlit UI with 3D WebGL visualizations
├── flask_app/                # RESTful API service with POST /api/analyze & web dashboard
├── configs/                  # Centralized YAML configuration files
├── models/                   # Serialized model artifacts & registry.json
├── results/                  # Metrics CSVs, confusion matrices, and experiment logs
├── tests/                    # Comprehensive unit, integration, and end-to-end tests
└── docs/                     # Technical, API, reproducibility, and ethical documentation
```

---

## 3. End-to-End Scientific Data Flow

```text
DNA / FASTA / FASTQ / VCF / CSV Input
                 ↓
[Stage 1]  Data Quality & IUPAC Integrity Audit (Strict ACGT vs Ambiguous IUPAC)
                 ↓
[Stage 2]  Reference Management & Validation (GRCh38 / Checksum / Reference-Blind Flag)
                 ↓
[Stage 3]  Sequence Alignment (Needleman-Wunsch Optimal Global DP)
                 ↓
[Stage 4]  Variant Extraction & Mutation Subtyping (SNV, Indel, Transition / Transversion)
                 ↓
[Stage 5]  Genomic Coordinate Normalization (GRCh38, 1-based, HGVS-c / HGVS-p)
                 ↓
[Stage 6]  Hybrid Feature Engineering (1-mer to 3-mer, Context, Entropy, and RKNP)
                 ↓
┌───────────────────────┬───────────────────────┬───────────────────────┐
↓                       ↓                       ↓                       ↓
Classical Baselines    TCN Deep Sequence       Quantum Classifiers     QMFN Hybrid Fusion
(LR, RF, SVM, GB)      (Dilated 1D Conv)       (VQC & QSVC Kernel)     (Dual-Branch Tensor)
└───────────────────────┴───────────────────────┴───────────────────────┘
                                    ↓
[Stage 7]  Champion & Ensemble Consensus Selection (Weighted Voting & Stacking)
                                    ↓
[Stage 8]  Model Agreement Scoring & Disagreement Warnings
                                    ↓
[Stage 9]  Genomic Annotation (Ensembl Transcripts & Canonical Registry)
                                    ↓
[Stage 10] External Biological Evidence Grounding (NCBI ClinVar with Provenance)
                                    ↓
[Stage 11] Multi-Modal Explainability (Gini Tabular, TCN Saliency, Quantum Sensitivity)
                                    ↓
[Stage 12] Conformal Uncertainty & Abstention (Finite-Sample Coverage Guarantees)
                                    ↓
[Stage 13] Unified VariantRecord & 14-Section Research Dossier Assembly
                                    ↓
Interactive Streamlit UI (Port 8501)  |  Flask REST API (Port 5000)
```

---

## 4. Dual-Branch QMFN Hybrid Architecture

The **Quantum Mutation Feature Network (QMFN)** addresses expressivity bottlenecks of standalone quantum circuits and the contextual limits of classical models:

```text
               DNA Features x ∈ R^D (K-mers, Entropy, Context, RKNP)
                                     │
                    ┌────────────────┴────────────────┐
                    ▼                                 ▼
         Classical Representation            Quantum State Mapping
                    │                                 │
         Dense Linear (D → 32)               PCA Projection (D → 4)
                    │                                 │
           LayerNorm + GELU                  Angle / ZZ Feature Map
                    │                                 │
         Dense Linear (32 → 16)              Parameterized Ansatz W(θ)
                    │                                 │
           LayerNorm + GELU                  Pauli-Z Measurements ⟨Z_i⟩
                    │                                 │
           h_classical ∈ R^16                 q_quantum ∈ [-1, 1]^4
                    │                                 │
                    └────────────────┬────────────────┘
                                     ▼
                        Bilinear Interaction Tensor
                            B_ij = h_i · q_j
                                     │
                     Concatenated Representation [h || q || vec(B)]
                                     │
                        Feature Fusion Layer (Dense)
                                     │
                            LayerNorm + GELU
                                     │
                           Residual Skip Connection
                                     │
                         Classification Head (Dense 2-way)
                                     │
                              Softmax Logits
```

---

## 5. Unified Data Contract (`VariantRecord`)

All components communicate through a single authoritative dataclass, [`src.variants.record.VariantRecord`](file:///c:/Users/Rajad/Downloads/DNA/src/variants/record.py), eliminating disparate dictionaries:

* **Identity:** `variant_id`, `sample_id`, `gene`, `chromosome`, `position`
* **Sequences:** `sequence_reference`, `sequence_alternate`, `normalized_reference`, `normalized_alternate`
* **Typology:** `mutation_type`, `mutation_subtype`, `hgvs_c`, `hgvs_p`, `consequence`
* **Predictions:** `prediction`, `probability`, `confidence`, `model_predictions`
* **Consensus:** `model_agreement_score`, `model_agreement_percentage`, `champion_model`
* **Uncertainty:** `conformal_prediction_set`, `conformal_uncertainty_level`, `conformal_abstention`
* **Evidence:** `evidence`, `evidence_source`, `evidence_provenance`, `evidence_confidence`
* **Attribution:** `top_contributing_features`, `explanation_summary`

---

## 6. Zero-Leakage Protocol

To ensure reproducible academic validity, data splitting follows three strict rules:
1. **Group-Aware Splitting:** Samples from the same gene or locus are strictly segregated into training, validation, or testing partitions.
2. **K-mer Novelty Auditing:** Training and testing partitions are audited using the 5-vector leakage audit ([`src.evaluation.leakage_audit`](file:///c:/Users/Rajad/Downloads/DNA/src/evaluation/leakage_audit.py)) to guarantee zero duplicate sequence overlap and controlled Jaccard k-mer similarity.
3. **Reference Isolation:** When reference sequences are condition-encoded, reference identifiers are scrubbed from the input vector to prevent shortcut learning.

# DNA Variant Intelligence Platform (DNA_v2) — Dataset Specifications

## 1. Overview & Ethical Scientific Positioning

All benchmark datasets used in the core evaluation of DNA_v2 are **in-silico synthetic genomic sequences** modeled after canonical human oncogenes and tumor suppressors (GRCh38).

> **Scientific Declaration:**
> The benchmark dataset does **NOT** represent real patient clinical sequencing data. It was synthetically synthesized in-silico to evaluate algorithmic sequence modeling, feature conditioning, and quantum Hilbert-space encoding in a controlled, ground-truth-verifiable experimental setting.

---

## 2. Benchmark Dataset Profiles

### 2.1. Curated Synthetic Cancer Variant Callset (`synthetic_dna_variants.csv`)
* **Path:** `data/raw/synthetic_dna_variants.csv`
* **Version:** `2.0.0`
* **Sample Count:** 396 sequence records
* **Label Distribution:**
  * Class 1 (Variant / Mutation): 208 samples (52.5%)
  * Class 0 (Wildtype / Reference): 188 samples (47.5%)
* **Target Genes Covered:**
  * `TP53` (Tumor Protein P53, Chromosome 17)
  * `BRCA1` (Breast Cancer 1, Chromosome 17)
  * `EGFR` (Epidermal Growth Factor Receptor, Chromosome 7)
  * `BRAF` (B-Raf Proto-Oncogene, Chromosome 7)
  * `KRAS` (KRAS Proto-Oncogene, Chromosome 12)
  * `CFTR` (CF Transmembrane Conductance Regulator, Chromosome 7)

### 2.2. Edge-Case Diagnostic Callset (`edge_case_dataset.csv`)
Designed to evaluate boundary conditions and edge cases:
* Homopolymer poly(A) and poly(T) tracts (>8 repeats)
* High GC-islands (>75% GC content) and AT-rich regions (<25% GC content)
* Dinucleotide and trinucleotide microsatellite tandem repeats
* Extreme length disparities (15 bp up to 5,000 bp)
* Degenerate IUPAC nucleotide codes (`R`, `Y`, `S`, `W`, `K`, `M`, `B`, `D`, `H`, `V`, `N`)

---

## 3. Schema & Field Definitions

| Field | Type | Example | Description |
| :--- | :--- | :--- | :--- |
| `variant_id` | String | `VAR_TP53_0042` | Unique system identifier |
| `gene` | String | `TP53` | HGNC official gene symbol |
| `chromosome` | String | `chr17` | Chromosomal locus |
| `position` | Integer | `7577120` | 1-based genomic start position (GRCh38) |
| `reference` | String | `C` | Canonical reference allele / nucleotide |
| `alternate` | String | `T` | Alternate / variant allele |
| `sequence` | String | `ATGCGATCGATC...` | Full 5' → 3' DNA sequence string |
| `label` | Integer | `1` | Ground-truth class (0: Wildtype, 1: Mutation) |
| `mutation_type` | String | `SNV` | Categorical typology (`SNV`, `INSERTION`, `DELETION`, `DUPLICATION`, `DELINS`) |
| `mutation_subtype`| String | `Transition` | Biochemical subtyping (`Transition`, `Transversion`, `Frameshift`, etc.) |
| `hgvs` | String | `c.14C>T` | Standardized HGVS nomenclature |

---

## 4. Nucleotide Composition & Entropy Metrics

* **Sequence Length Range:** 20 bp to 120 bp (Mean: 54.2 bp, Std: 8.6 bp)
* **Mean GC Content:** 48.6% (Range: 38.2% to 62.4%)
* **Mean Shannon Entropy:** 1.942 bits / base (Theoretical maximum for uniform 4-base alphabet: $\log_2(4) = 2.0$ bits)
* **Ambiguous Nucleotides:** 0% in cleaned benchmark partitions

---

## 5. Splitting Strategy & Leakage Controls

Partitions are generated using **Group-Aware Stratified Splitting**:
1. **Training Partition (70%):** Model parameter optimization and feature extraction fitting.
2. **Validation Partition (15%):** Hyperparameter tuning, early stopping, and stacking meta-learner calibration.
3. **Test Partition (15%):** Held-out final benchmark evaluation.

Zero sequence overlap exists between partitions. Leave-One-Gene-Out cross-validation is additionally implemented to evaluate cross-gene generalization.

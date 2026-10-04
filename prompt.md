You are an expert Machine Learning, Deep Learning, Quantum Machine Learning, Bioinformatics, Python, and Streamlit developer.

I am developing an M.E. Computer Science research project titled:

"A Hybrid Classical–Quantum Framework for DNA Variant Detection, Classification and Evidence-Based Biological Interpretation"

Build this project as a complete, modular, research-oriented Python application.

IMPORTANT:

* Do not fabricate biological or clinical results.
* Do not claim that a mutation means a patient has a disease.
* Disease association must be based on external validated biological resources and/or literature.
* Clearly separate ML prediction from biological/database evidence.
* Do not claim that Quantum ML is better than Classical ML before experimental evaluation.
* All model comparisons must be based on measured results.
* Use reproducible experiments with fixed random seeds.
* Keep all intermediate results and final metrics saved to files.
* Use clear comments and simple code because this project will be used for an M.E. research thesis.

==================================================

1. PROJECT OBJECTIVE
   ==================================================

Develop a complete pipeline:

DNA Dataset
→ Data Validation
→ DNA Preprocessing
→ Feature Engineering
→ Classical ML
→ TCN Deep Learning
→ Quantum ML
→ Proposed QMFN
→ Mutation Detection
→ Mutation Classification
→ Mutation Localization
→ Variant Normalization
→ Genomic Annotation
→ Biological Interpretation
→ Disease/Trait Association
→ Evidence Integration
→ Explainable AI
→ Statistical Validation
→ Classical vs Deep Learning vs Quantum Comparison
→ Final Mutation Report
→ Streamlit Dashboard

==================================================
2. RECOMMENDED PROJECT STRUCTURE
================================

Create:

dna_variant_project/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── models/
│   ├── classical/
│   ├── deep_learning/
│   ├── quantum/
│   └── qmfnet/
│
├── src/
│   ├── preprocessing/
│   ├── features/
│   ├── classical/
│   ├── deep_learning/
│   ├── quantum/
│   ├── qmfnet/
│   ├── mutation/
│   ├── annotation/
│   ├── disease_association/
│   ├── explainability/
│   ├── evaluation/
│   └── reporting/
│
├── dashboard/
│   └── app.py
│
├── notebooks/
│
├── results/
│   ├── metrics/
│   ├── predictions/
│   ├── figures/
│   └── reports/
│
├── tests/
│
├── requirements.txt
├── config.yaml
├── README.md
└── train.py

==================================================
3. DATA INPUT
=============

Support CSV input.

Minimum recommended fields:

sequence
label

If available, also support:

variant_id
chromosome
position
reference
alternate
gene
transcript
mutation_type
condition
hgvs

Do not assume every dataset contains all fields.

The application must inspect the uploaded dataset and dynamically determine which columns are available.

Display:

* Number of records
* Number of columns
* Sequence length statistics
* Class distribution
* Missing values
* Duplicate records
* Invalid nucleotide characters

==================================================
4. DNA PREPROCESSING
====================

Implement:

* Convert sequence to uppercase
* Validate A/T/G/C characters
* Handle N or ambiguous bases
* Remove or flag invalid sequences
* Remove duplicates
* Handle missing values
* Check sequence length
* Encode labels
* Detect class imbalance
* Train/validation/test split

Avoid data leakage.

For sequence datasets, make sure related or duplicated sequences do not appear across both training and test sets when the dataset structure requires grouping.

Save:

processed_dataset.csv

==================================================
5. DNA FEATURE ENGINEERING
==========================

Implement:

A. Basic features

* Sequence length
* A frequency
* T frequency
* G frequency
* C frequency
* GC content
* AT content

B. K-mer features

Support:

* 2-mer
* 3-mer
* 4-mer

Make k configurable.

C. Sequence statistics

* Shannon entropy
* nucleotide diversity
* GC/AT ratio

D. Mutation-context features, when reference and alternate sequences are available

* mutation position
* reference nucleotide
* alternate nucleotide
* local sequence window
* surrounding nucleotide composition

Save the extracted feature matrix.

==================================================
6. CLASSICAL MACHINE LEARNING
=============================

Implement these baseline models:

1. Logistic Regression
2. Random Forest
3. SVM
4. KNN
5. Gradient Boosting

Use:

* Stratified cross-validation
* Hyperparameter configuration
* Random seeds
* Standard preprocessing pipelines where appropriate

Calculate:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC where applicable
* PR-AUC where applicable
* Confusion matrix
* Training time
* Inference time

Save all results to:

results/metrics/classical_results.csv

==================================================
7. TCN DEEP LEARNING MODEL
==========================

Implement a Temporal Convolutional Network for DNA sequence classification.

Architecture should contain:

Input DNA sequence
→ nucleotide encoding
→ embedding/feature representation
→ causal/dilated convolutions
→ residual blocks
→ global pooling
→ dense layer
→ mutation prediction

Use configurable:

* number of filters
* kernel size
* dilation rates
* dropout
* learning rate
* epochs
* batch size

Include:

* Early stopping
* Validation monitoring
* Model checkpointing

Save:

* trained model
* training history
* validation metrics
* test predictions

==================================================
8. QUANTUM MACHINE LEARNING
===========================

Implement two quantum baselines:

A. Variational Quantum Classifier (VQC)

Pipeline:

DNA Features
→ dimensionality reduction
→ normalization
→ quantum feature encoding
→ parameterized quantum circuit
→ measurement
→ classifier

B. Quantum Kernel

Pipeline:

DNA Features
→ dimensionality reduction
→ quantum encoding
→ quantum kernel
→ kernel classifier
→ mutation prediction

Use a simulator by default so the project can run without quantum hardware.

Keep the number of qubits configurable.

Record:

* number of qubits
* circuit depth
* encoding method
* number of parameters
* training time
* prediction time
* evaluation metrics

IMPORTANT:
Do not claim quantum advantage unless the experiments demonstrate it under a clearly defined comparison.

==================================================
9. PROPOSED QMFN
================

Implement the proposed:

"Quantum Mutation Feature Network (QMFN)"

Architecture:

DNA Features
↓
┌───────────────┐
│ Classical     │
│ Feature Branch│
└───────┬───────┘
│
├──────────────┐
│              ↓
│        Feature Fusion
│              ↑
┌───────┴────────┐     │
│ Quantum        │─────┘
│ Feature Branch │
└───────┬────────┘
↓
Quantum Encoding
↓
Variational Quantum Circuit
↓
Quantum Measurements
↓
Feature Fusion
↓
Classical Output Layer
↓
Mutation Prediction

Make the architecture modular.

Document clearly which components are proposed by this project.

Do not describe QMFN as a proven novel algorithm unless novelty has been established through a literature review.

==================================================
10. MUTATION DETECTION
======================

Create a dedicated module.

Input:

DNA sequence or reference/alternate sequence

Output:

Mutation Detected = YES / NO

For probability-based models:

Mutation Probability = X

Do not confuse model confidence with biological certainty.

==================================================
11. MUTATION CLASSIFICATION
===========================

If mutation is detected, classify:

* Substitution
* Insertion
* Deletion
* Duplication

If the dataset contains other mutation types, preserve them instead of forcing them into these four categories.

Output:

Mutation Type
Prediction Probability
Model Used

==================================================
12. MUTATION LOCALIZATION
=========================

When reference and alternate sequences are available, identify:

* mutation position
* reference base
* alternate base
* sequence context

Example:

Reference:
ATGCCATGGA

Alternate:
ATGCTATGGA

Output:

Position = 5
Reference = C
Alternate = T
Change = C>T

If localization cannot be reliably calculated from the available data, display:

"Localization unavailable from supplied sequence information."

Never invent a position.

==================================================
13. VARIANT NORMALIZATION
=========================

Implement a normalization layer before external annotation.

Capture:

* reference genome assembly
* chromosome
* genomic coordinate
* reference allele
* alternate allele
* transcript
* HGVS when available
* existing variant identifier when available

The system must explicitly show which fields are experimentally supplied and which are derived.

==================================================
14. GENOMIC ANNOTATION
======================

Where genomic coordinates are available, support annotation using appropriate bioinformatics resources/tools.

Retrieve, where available:

* gene
* transcript
* exon/intron
* coding region
* UTR
* regulatory region
* molecular consequence
* codon change
* amino-acid change

Do not infer genomic coordinates from a short DNA sequence unless a valid reference mapping is available.

==================================================
15. BIOLOGICAL INTERPRETATION
=============================

Create:

src/annotation/biological_interpretation.py

Analyze:

A. Genomic context
B. Gene
C. Transcript
D. Coding/non-coding status
E. Molecular consequence
F. Codon change
G. Amino-acid change
H. Protein consequence
I. Conservation where available
J. Population frequency where available
K. Biological pathway where available

Separate:

MODEL PREDICTION

from

DATABASE/LITERATURE EVIDENCE

==================================================
16. DISEASE / TRAIT ASSOCIATION
===============================

Create:

src/disease_association/

The module must NOT diagnose disease.

Its purpose is:

"Determine whether the specific variant has a previously reported association with a disease or trait."

Use validated sources such as:

* ClinVar
* Ensembl/VEP
* gnomAD or another appropriate population resource
* PubMed
* ClinGen where applicable

Workflow:

Predicted Variant
→ Standardized Variant
→ Database Search
→ Retrieve Variant-Condition Evidence
→ Retrieve Classification
→ Retrieve Review Status
→ Retrieve Literature
→ Retrieve Population Evidence
→ Evidence Integration

Output one of:

1. REPORTED_ASSOCIATION
2. NO_REPORTED_ASSOCIATION_FOUND
3. CONFLICTING_EVIDENCE
4. INSUFFICIENT_EVIDENCE

Never output:

"Patient has disease."

Instead output:

"Known Disease Association: YES"

and provide the referenced condition and evidence source.

==================================================
17. EVIDENCE TABLE
==================

Generate:

Evidence Source
Result
Classification
Condition
Review Status
Publication
Population Frequency
Evidence Notes

Example:

Source: ClinVar
Status: Found
Condition: [database result]
Classification: [database result]
Review Status: [database result]

Never fabricate missing information.

==================================================
18. EXPLAINABLE AI
==================

Implement model explainability where technically appropriate.

For classical tabular models:

* SHAP
* permutation importance
* feature importance

For TCN:

* sequence/feature attribution method where appropriate

For quantum models:

* circuit expectation/feature sensitivity analysis where technically justified

Visualize:

* important features
* mutation-context contribution
* model prediction probability
* comparison of model explanations

Clearly label computational explanations as model explanations, not biological proof.

==================================================
19. MODEL COMPARISON
====================

Compare:

Logistic Regression
Random Forest
SVM
KNN
Gradient Boosting
TCN
VQC
Quantum Kernel
QMFN

Metrics:

Accuracy
Precision
Recall
F1
ROC-AUC
PR-AUC
Training Time
Inference Time

Generate:

* comparison table
* bar charts
* confusion matrices
* ROC curves
* PR curves

Do not automatically label any model "best."

Determine the result from the test-set metrics and statistical analysis.

==================================================
20. STATISTICAL VALIDATION
==========================

Implement:

* stratified cross-validation
* repeated experiments
* mean
* standard deviation
* confidence intervals where appropriate

For model comparisons, use an appropriate statistical test based on the experimental design.

Document:

* test used
* null hypothesis
* significance level
* interpretation

==================================================
21. FINAL MUTATION REPORT
=========================

Generate a structured report:

---

## DNA VARIANT ANALYSIS REPORT

Input Sequence:
...

Mutation Detected:
YES / NO

Mutation Type:
...

Mutation Position:
...

Reference Base:
...

Alternate Base:
...

Model:
...

Prediction Probability:
...

---

## GENOMIC ANNOTATION

Chromosome:
...

Gene:
...

Transcript:
...

Region:
...

Molecular Consequence:
...

Codon Change:
...

Amino Acid Change:
...

---

## BIOLOGICAL INTERPRETATION

Conservation:
...

Population Frequency:
...

Biological Function:
...

Pathway:
...

---

## DISEASE / TRAIT ASSOCIATION

Known Association:
YES / NO / UNCERTAIN

Condition:
...

Evidence Source:
...

Classification:
...

Review Status:
...

Literature:
...

Evidence Notes:
...

---

## MODEL COMPARISON

Classical:
...

TCN:
...

Quantum:
...

QMFN:
...

---

DISCLAIMER

This computational analysis is not a medical diagnosis.
Disease/trait association is reported only from available
referenced evidence and should not be interpreted as
patient-specific clinical diagnosis.
------------------------------------

==================================================
22. STREAMLIT DASHBOARD
=======================

Build a professional Streamlit dashboard.

Pages/tabs:

1. Home
2. Dataset
3. Data Quality
4. DNA Preprocessing
5. Feature Engineering
6. Classical ML
7. TCN
8. Quantum ML
9. QMFN
10. Mutation Detection
11. Mutation Classification
12. Mutation Localization
13. Variant Annotation
14. Biological Interpretation
15. Disease Association
16. Explainable AI
17. Model Comparison
18. Final Mutation Report

Use:

* tables
* metric cards
* DNA sequence highlighting
* mutation position visualization
* confusion matrices
* ROC/PR plots
* model comparison charts
* evidence tables

==================================================
23. DATABASE/API DESIGN
=======================

Create separate connectors/modules for external biological resources.

Do not hard-code disease results.

Implement:

* API requests
* timeout handling
* retry handling
* response validation
* caching
* logging
* rate-limit handling
* source timestamps

Store retrieved evidence locally when appropriate so experiments can be reproduced.

==================================================
24. REPRODUCIBILITY
===================

Create:

config.yaml

Include:

* random seed
* test size
* validation size
* k-mer size
* model parameters
* quantum qubit count
* circuit depth
* training epochs
* batch size

Every experiment must save its configuration.

==================================================
25. TESTING
===========

Create unit tests for:

* sequence validation
* GC calculation
* k-mer generation
* mutation localization
* mutation classification
* variant normalization
* API response parsing
* disease association logic
* report generation

Also create an end-to-end test using a small synthetic dataset.

Synthetic data must be clearly labeled as synthetic.

==================================================
26. README
==========

Create a complete README containing:

Project Overview
Research Problem
Objectives
Architecture
Installation
Dataset Format
Preprocessing
Classical ML
TCN
Quantum ML
QMFN
Mutation Detection
Mutation Classification
Variant Annotation
Disease Association
Explainable AI
Evaluation
Dashboard
Limitations
Reproducibility
Ethical/Clinical Disclaimer

==================================================
27. DEVELOPMENT RULE
====================

Do NOT generate the entire application in one huge file.

Implement one module at a time.

After each module:

1. Explain what was implemented.
2. Show the files created/modified.
3. Run tests.
4. Fix errors.
5. Continue to the next module.

Start with:

Phase 1:
Project structure + configuration + dataset loader + validation + preprocessing.

Then wait for confirmation before implementing the next major phase.

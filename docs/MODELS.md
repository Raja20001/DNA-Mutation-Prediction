# DNA Variant Intelligence Platform (DNA_v2) — Model Specifications

## 1. Architectural Taxonomy

DNA_v2 implements a multi-paradigm benchmark evaluating 9 distinct model architectures across four families:
1. **Classical Machine Learning Baselines** (Tabular K-mer & Biophysical Features)
2. **Deep Sequence Learning** (Dilated Causal 1D Convolutions on Token Tensors)
3. **Quantum Machine Learning** (Hilbert-Space Parameterized Circuits & Quantum Kernels)
4. **Hybrid Classical–Quantum Architecture (QMFN)** (Dual-Branch Bilinear Tensor Fusion)

---

## 2. Model Architecture Profiles

### 2.1. Classical Baselines

| Model | Hyperparameters / Topology | Train Time (s) | Inference (s) | Best F1 |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | L2 penalty ($C=1.0$), `lbfgs` solver, max_iter=500 | 0.08s | <0.001s | 0.850 |
| **Random Forest** | 100 trees, Gini impurity, max_depth=12, min_samples_split=4 | 0.42s | 0.012s | 0.933 |
| **Support Vector Classifier (SVM)** | RBF kernel ($\gamma=\text{scale}$), $C=1.0$, Platt probability scaling | 0.15s | 0.005s | 0.901 |
| **K-Nearest Neighbors (KNN)** | $K=5$, Euclidean metric, uniform weights | 0.04s | 0.003s | 0.817 |
| **Gradient Boosting** | 100 boosting stages, learning_rate=0.10, max_depth=3 | 0.65s | 0.008s | 0.917 |

---

### 2.2. Temporal Convolutional Network (TCN Deep Learning)

* **Architecture:** 1D Causal Dilated Residual Convolutional Network
* **Input Representation:** DNA Token Indices ($\text{A}=1, \text{C}=2, \text{G}=3, \text{T}=4, \text{Pad}=0$)
* **Embedding Layer:** Vocabulary=5, Embedding Dimension=16
* **Convolutional Channels:** 64 filters per residual temporal block
* **Dilation Schedule:** $d \in [1, 2, 4, 8]$ (Receptive field exceeds 32 bp)
* **Kernel Size:** $k = 3$ with causal padding
* **Pooling:** Adaptive Average Pooling 1D (collapsing sequence length to 1)
* **Classification Head:** Linear (64 → 32) → ReLU → Dropout (0.2) → Linear (32 → 2) → Softmax
* **Total Trainable Parameters:** 52,866
* **Test Performance:** Accuracy: 0.950, F1-Score: 0.950, ROC-AUC: 0.985

---

### 2.3. Quantum Machine Learning (Qiskit Statevector Simulator)

#### 2.3.1. Variational Quantum Classifier (VQC)
* **Qubits:** 4 entangled qubits (Hilbert Space Dimension $\dim(\mathcal{H}) = 2^4 = 16$)
* **Feature Map:** Second-Order Pauli-Z (`ZZFeatureMap`, Repetitions=2, full linear entanglement)
* **Variational Ansatz:** `RealAmplitudes` (Alternating single-qubit $R_y(\theta)$ rotations and CNOT gates)
* **Circuit Depth:** 18 gates
* **Classical Optimizer:** COBYLA (maxiter=50, tolerance=$10^{-4}$)
* **Performance:** Accuracy: 0.867, F1-Score: 0.865, Train Time: 18.4s

#### 2.3.2. Quantum Support Vector Classifier (QSVC / Quantum Kernel)
* **Kernel Formulation:** Quantum state fidelity $K_{ij} = |\langle \Phi(\mathbf{x}_i) | \Phi(\mathbf{x}_j) \rangle|^2$
* **Feature Map:** Non-linear ZZ-feature map creating block-diagonal separation in $\mathcal{H}$
* **Classifier:** Dual quadratic program solver with precomputed quantum Gram matrix
* **Performance:** Accuracy: 0.900, F1-Score: 0.898, Train Time: 6.2s

#### 2.3.3. Quantum Convolutional Neural Network (QCNN)
* **Ansatz:** Alternating 2-qubit unitary convolutions $U_C$ and entangled measurement pooling $U_P$
* **Barren Plateau Mitigation:** Logarithmic circuit depth $\mathcal{O}(\log N)$ preserving gradient variance
* **Trainable Parameters:** 16 rotation parameters

---

### 2.4. Proposed QMFN Hybrid Architecture

The **Quantum Mutation Feature Network (QMFN)** introduces dual-branch tensor interaction:

1. **Classical Branch:**
   $$\mathbf{h}_1 = \text{GELU}(\text{LayerNorm}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)), \quad \mathbf{W}_1 \in \mathbb{R}^{32 \times D}$$
   $$\mathbf{h}_{\text{classical}} = \text{GELU}(\text{LayerNorm}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2)), \quad \mathbf{h}_{\text{classical}} \in \mathbb{R}^{16}$$

2. **Quantum Branch:**
   Continuous features $\mathbf{x}$ projected via PCA to $\mathbb{R}^4$, angle-encoded into 4 qubits:
   $$\mathbf{q}_{\text{quantum}} = \left[ \langle Z_1 \rangle, \langle Z_2 \rangle, \langle Z_3 \rangle, \langle Z_4 \rangle \right]^T \in [-1, 1]^4$$

3. **Bilinear Interaction Core:**
   $$\mathbf{B} = \mathbf{h}_{\text{classical}} \otimes \mathbf{q}_{\text{quantum}} \in \mathbb{R}^{16 \times 4}$$
   $$\mathbf{z}_{\text{fused}} = \text{GELU}\left(\text{LayerNorm}\left(\mathbf{W}_f [\mathbf{h}_{\text{classical}} \parallel \mathbf{q}_{\text{quantum}} \parallel \text{vec}(\mathbf{B})] + \mathbf{b}_f\right)\right)$$

4. **Performance & Ablation:**
   * **Accuracy:** 0.967
   * **F1-Score:** 0.967
   * **ROC-AUC:** 0.990
   * **Brier Score:** 0.041 (Exceptional probability calibration)

---

## 3. Scientific Honesty & Positioning

* **No Quantum Advantage Claim:** The quantum simulator demonstrates the viability of high-dimensional non-linear state mapping for genomic motifs, but does **not** claim computational quantum supremacy over classical hardware.
* **Classical Baseline Strength:** Well-tuned tree ensembles (Random Forest, Gradient Boosting) achieve strong results (~0.93 F1) with negligible computational overhead (0.42s vs 18.4s for VQC).

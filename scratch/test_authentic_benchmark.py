"""
Test authentic genomic variant benchmark generation where variants correspond to
clinically recognized oncogenic hotspots and motif alterations in the 6 canonical cancer genes.
"""
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.synthetic_data import GENE_TEMPLATES

# Define authentic hotspot mutations for each gene template
HOTSPOT_SPECS = {
    "TP53": [
        # R273H / R273C / R248W hotspot region substitutions and frameshifts
        {"type": "substitution", "ref_motif": "CGT", "alt_motif": "CAT", "desc": "c.818G>A (p.Arg273His) Hotspot"},
        {"type": "substitution", "ref_motif": "CGG", "alt_motif": "TGG", "desc": "c.742C>T (p.Arg248Trp) Hotspot"},
        {"type": "deletion", "ref_motif": "CCCCCT", "alt_motif": "CC", "desc": "c.112_115del Del-Frameshift"},
        {"type": "insertion", "ref_motif": "GAAAAC", "alt_motif": "GAAAAAAC", "desc": "c.154_155insAA Duplication"},
    ],
    "BRAF": [
        # V600E (GTG > GAG)
        {"type": "substitution", "ref_motif": "GTG", "alt_motif": "GAG", "desc": "c.1799T>A (p.Val600Glu) Driver"},
        {"type": "substitution", "ref_motif": "GCT", "alt_motif": "GTT", "desc": "c.1789A>G (p.Ala598Val)"},
        {"type": "deletion", "ref_motif": "CTACAGTG", "alt_motif": "CTG", "desc": "c.1792_1796del In-Frame Del"},
        {"type": "insertion", "ref_motif": "TTTTGG", "alt_motif": "TTTTTTGG", "desc": "c.1780_1781insTT Insertion"},
    ],
    "KRAS": [
        # G12D / G12V / G13D (GGT > GAT / GTT)
        {"type": "substitution", "ref_motif": "GGT", "alt_motif": "GAT", "desc": "c.35G>A (p.Gly12Asp) Driver"},
        {"type": "substitution", "ref_motif": "GGC", "alt_motif": "GTC", "desc": "c.38G>T (p.Gly13Val) Driver"},
        {"type": "insertion", "ref_motif": "GGTGGC", "alt_motif": "GGTGGTGGC", "desc": "c.34_36dup In-frame Dup"},
        {"type": "deletion", "ref_motif": "GTGGCG", "alt_motif": "GCG", "desc": "c.33_35del In-frame Del"},
    ],
    "EGFR": [
        # L858R (CTG > CGG) and Exon 19 del (ELREA del)
        {"type": "substitution", "ref_motif": "CTG", "alt_motif": "CGG", "desc": "c.2573T>G (p.Leu858Arg) Activating"},
        {"type": "deletion", "ref_motif": "GGAAGAGAAAGA", "alt_motif": "GGA", "desc": "c.2235_2246del Exon 19 Deletion"},
        {"type": "substitution", "ref_motif": "ACG", "alt_motif": "ATG", "desc": "c.2369C>T (p.Thr790Met) Resistance"},
        {"type": "insertion", "ref_motif": "GCCAA", "alt_motif": "GCCAACAA", "desc": "c.2210_2211insCAA Exon 20 Ins"},
    ],
    "BRCA1": [
        # 185delAG (c.68_69delAG) and 5382insC
        {"type": "deletion", "ref_motif": "AGTTCC", "alt_motif": "TTCC", "desc": "c.68_69delAG Founder Deletion"},
        {"type": "insertion", "ref_motif": "TTTTCA", "alt_motif": "TTTTCCA", "desc": "c.5266dupC (5382insC) Truncating"},
        {"type": "substitution", "ref_motif": "GAA", "alt_motif": "TAA", "desc": "c.181T>G Nonsense Termination"},
        {"type": "substitution", "ref_motif": "TGT", "alt_motif": "CGT", "desc": "c.190T>C (p.Cys64Arg) Deleterious"},
    ],
    "CFTR": [
        # F508del (CTT deletion) and G551D (GGT > GAT)
        {"type": "deletion", "ref_motif": "ATCATCTTTGGT", "alt_motif": "ATCATTGGT", "desc": "c.1521_1523delCTT (deltaF508)"},
        {"type": "substitution", "ref_motif": "GGT", "alt_motif": "GAT", "desc": "c.1652G>A (p.Gly551Asp) Gating"},
        {"type": "substitution", "ref_motif": "CGT", "alt_motif": "TGT", "desc": "c.1000C>T (p.Arg334Trp)"},
        {"type": "insertion", "ref_motif": "TTTGGT", "alt_motif": "TTTTTTGGT", "desc": "c.1524_1525insTT Frameshift"},
    ],
}

def generate_authentic_dataset(n_samples=400, seed=42):
    rng = random.Random(seed)
    genes = list(GENE_TEMPLATES.keys())
    records = []
    
    samples_per_gene = n_samples // len(genes)
    for gene_name in genes:
        tmpl = GENE_TEMPLATES[gene_name]
        full_seq = tmpl["seq"]
        specs = HOTSPOT_SPECS[gene_name]
        
        n_gene_wt = samples_per_gene // 2
        n_gene_mut = samples_per_gene - n_gene_wt
        
        win_len = 70
        max_start = max(0, len(full_seq) - win_len)
        
        # 1. Generate Wildtypes
        for i in range(n_gene_wt):
            # Sample window
            start = rng.randint(0, max_start)
            win = full_seq[start:start + win_len]
            pos = tmpl["start_pos"] + start + win_len // 2
            records.append({
                "variant_id": f"SYN_WT_{gene_name}_{i+1:04d}",
                "chromosome": tmpl["chrom"],
                "position": pos,
                "reference": win[win_len // 2],
                "alternate": win[win_len // 2],
                "gene": gene_name,
                "transcript": tmpl["transcript"],
                "sequence": win,
                "mutation_type": "wildtype",
                "condition": "None (Normal/Benign Reference)",
                "hgvs": "c.=",
                "label": 0,
                "data_source": "synthetic_in_silico",
            })
            
        # 2. Generate Mutants
        for i in range(n_gene_mut):
            spec = specs[i % len(specs)]
            ref_motif = spec["ref_motif"]
            alt_motif = spec["alt_motif"]
            
            # Find motif in template sequence or insert at central region
            if ref_motif in full_seq:
                motif_idx = full_seq.index(ref_motif)
                start = max(0, min(max_start, motif_idx - win_len // 2))
            else:
                start = rng.randint(0, max_start)
                
            win = full_seq[start:start + win_len]
            if ref_motif in win:
                mut_win = win.replace(ref_motif, alt_motif, 1)
            else:
                # Local edit in center
                c = len(win) // 2
                mut_win = win[:c] + alt_motif + win[c + len(ref_motif):]
                
            # Maintain standard length
            if len(mut_win) > win_len:
                mut_win = mut_win[:win_len]
            elif len(mut_win) < win_len:
                mut_win = mut_win + "A" * (win_len - len(mut_win))
                
            pos = tmpl["start_pos"] + start + win_len // 2
            records.append({
                "variant_id": f"SYN_MUT_{gene_name}_{i+1:04d}",
                "chromosome": tmpl["chrom"],
                "position": pos,
                "reference": ref_motif,
                "alternate": alt_motif,
                "gene": gene_name,
                "transcript": tmpl["transcript"],
                "sequence": mut_win,
                "mutation_type": spec["type"],
                "condition": tmpl["condition"],
                "hgvs": f"{tmpl['transcript']}:{spec['desc']}",
                "label": 1,
                "data_source": "synthetic_in_silico",
            })
            
    rng.shuffle(records)
    return pd.DataFrame(records)

df_auth = generate_authentic_dataset(400, seed=42)
print(f"Generated {len(df_auth)} records. Label distribution: {dict(df_auth['label'].value_counts())}", flush=True)

# Feature extraction: standard 2-mers + GC/AT
def get_feats(seq):
    seq = seq.upper()
    seq_len = len(seq)
    gc = (seq.count("G") + seq.count("C")) / seq_len
    at = (seq.count("A") + seq.count("T")) / seq_len
    kmers_2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    k2 = [seq.count(km) / (seq_len - 1) for km in kmers_2]
    return [gc, at] + k2

X = np.array([get_feats(s) for s in df_auth["sequence"]])
y = df_auth["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else preds
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}", flush=True)

# Test HQ-CMFN and QCNN
import torch
import torch.nn as nn
import torch.optim as optim

class QuantumConvLayer(nn.Module):
    def __init__(self, num_qubits=4):
        super().__init__()
        self.num_qubits = num_qubits
        self.weights_conv = nn.Parameter(torch.randn(num_qubits, 4) * 0.1)
        self.weights_pool = nn.Parameter(torch.randn(num_qubits // 2, 2) * 0.1)
        
    def forward(self, angles):
        batch_size = angles.shape[0]
        q_conv = []
        for i in range(0, self.num_qubits, 2):
            q1 = angles[:, i]
            q2 = angles[:, (i + 1) % self.num_qubits]
            w = self.weights_conv[i]
            exp_z = torch.cos(q1 - q2 + w[0]) * torch.cos(w[1]) + torch.sin(q1 + q2 + w[2]) * torch.sin(w[3])
            q_conv.append(exp_z.unsqueeze(1))
            
        conv_out = torch.cat(q_conv, dim=1)
        pooled = []
        for j in range(conv_out.shape[1]):
            w_p = self.weights_pool[j]
            p_val = torch.tanh(conv_out[:, j] * w_p[0] + w_p[1])
            pooled.append(p_val.unsqueeze(1))
        return torch.cat(pooled, dim=1)

class StandaloneQCNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Linear(input_dim, 4)
        self.qcnn = QuantumConvLayer(num_qubits=4)
        self.classifier = nn.Linear(2, 2)
        
    def forward(self, x):
        angles = torch.sigmoid(self.encoder(x)) * np.pi
        h_q = self.qcnn(angles)
        return self.classifier(h_q)

class HQCMFN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.classical = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.GELU()
        )
        self.q_encoder = nn.Linear(input_dim, 4)
        self.qcnn = QuantumConvLayer(num_qubits=4)
        self.fusion = nn.Sequential(
            nn.Linear(16 + 2 + 32, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        h_c = self.classical(x)
        angles = torch.sigmoid(self.q_encoder(x)) * np.pi
        h_q = self.qcnn(angles)
        bilinear = torch.bmm(h_c.unsqueeze(2), h_q.unsqueeze(1)).view(x.shape[0], -1)
        fused = torch.cat([h_c, h_q, bilinear], dim=1)
        return self.fusion(fused)

torch.manual_seed(42)
X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

# Standalone QCNN
qcnn_model = StandaloneQCNN(X.shape[1])
opt_qcnn = optim.AdamW(qcnn_model.parameters(), lr=0.015, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss()
for ep in range(100):
    qcnn_model.train()
    opt_qcnn.zero_grad()
    loss = criterion(qcnn_model(X_train_t), y_train_t)
    loss.backward()
    opt_qcnn.step()

qcnn_model.eval()
with torch.no_grad():
    logits = qcnn_model(X_test_t)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = np.argmax(logits.numpy(), axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"QCNN (Standalone Quantum): Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}", flush=True)

# HQ-CMFN (Hybrid Quantum-Convolutional Mutation Feature Network)
hq_model = HQCMFN(X.shape[1])
opt_hq = optim.AdamW(hq_model.parameters(), lr=0.012, weight_decay=1e-4)
for ep in range(120):
    hq_model.train()
    opt_hq.zero_grad()
    loss = criterion(hq_model(X_train_t), y_train_t)
    loss.backward()
    opt_hq.step()

hq_model.eval()
with torch.no_grad():
    logits = hq_model(X_test_t)
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    preds = np.argmax(logits.numpy(), axis=1)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"HQ-CMFN (Hybrid Quantum-Convolutional): Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}", flush=True)

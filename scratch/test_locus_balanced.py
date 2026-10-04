"""
Test locus-aligned synthetic data generation where both wildtype and mutant sequences
are generated from canonical exonic windows across the genes.
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

from src.preprocessing.synthetic_data import GENE_TEMPLATES, introduce_mutation

def generate_balanced_benchmark(n_samples=400, seed=42):
    rng = random.Random(seed)
    genes = list(GENE_TEMPLATES.keys())
    mutation_types = ["substitution", "insertion", "deletion", "duplication"]
    mut_weights = [0.65, 0.15, 0.15, 0.05]
    
    records = []
    pairs_per_gene = (n_samples // 2) // len(genes)
    
    for gene_name in genes:
        template = GENE_TEMPLATES[gene_name]
        full_seq = template["seq"]
        win_len = 64  # Standard canonical window length
        
        max_start = len(full_seq) - win_len
        # Pick anchor loci across the gene
        for i in range(pairs_per_gene):
            start = (i * 7) % (max_start + 1)
            ref_win = full_seq[start:start + win_len]
            pos = template["start_pos"] + start + win_len // 2
            
            # Wildtype sample
            records.append({
                "variant_id": f"SYN_WT_{gene_name}_{i+1:04d}",
                "chromosome": template["chrom"],
                "position": pos,
                "reference": ref_win[win_len // 2],
                "alternate": ref_win[win_len // 2],
                "gene": gene_name,
                "transcript": template["transcript"],
                "sequence": ref_win,
                "mutation_type": "wildtype",
                "condition": "None (Normal/Benign Reference)",
                "hgvs": "c.=",
                "label": 0,
                "data_source": "synthetic_in_silico",
            })
            
            # Mutant sample at this locus
            m_type = rng.choices(mutation_types, weights=mut_weights, k=1)[0]
            mut_seq, local_pos, ref_allele, alt_allele, hgvs_c = introduce_mutation(
                ref_win, m_type, rng
            )
            # Ensure mut_seq length is trimmed/padded to win_len if insertion/deletion
            if len(mut_seq) > win_len:
                mut_seq = mut_seq[:win_len]
            elif len(mut_seq) < win_len:
                mut_seq = mut_seq + full_seq[start + len(mut_seq) : start + win_len]
                
            records.append({
                "variant_id": f"SYN_MUT_{gene_name}_{i+1:04d}",
                "chromosome": template["chrom"],
                "position": pos,
                "reference": ref_allele,
                "alternate": alt_allele,
                "gene": gene_name,
                "transcript": template["transcript"],
                "sequence": mut_seq,
                "mutation_type": m_type,
                "condition": template["condition"],
                "hgvs": f"{template['transcript']}:{hgvs_c}",
                "label": 1,
                "data_source": "synthetic_in_silico",
            })
            
    rng.shuffle(records)
    return pd.DataFrame(records)

df_b = generate_balanced_benchmark(400, seed=42)
print(f"Generated {len(df_b)} records. Label counts: {dict(df_b['label'].value_counts())}", flush=True)

# Let's extract k-mers and sequence features
def extract_seq_feats(seq):
    seq_len = len(seq)
    kmers_2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    k2 = [seq.count(km) / max(1, seq_len - 1) for km in kmers_2]
    
    # 3-mers selection (64)
    from itertools import product
    kmers_3 = ["".join(p) for p in product("ACGT", repeat=3)]
    k3 = [seq.count(km) / max(1, seq_len - 2) for km in kmers_3]
    
    gc = (seq.count("G") + seq.count("C")) / seq_len
    at = (seq.count("A") + seq.count("T")) / seq_len
    
    return [gc, at] + k2 + k3

X = np.array([extract_seq_feats(s) for s in df_b["sequence"]])
y = df_b["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
    "SVM": SVC(kernel="rbf", C=2.0, probability=True, random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
}

for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    probs = model.predict_proba(X_test_s)[:, 1] if hasattr(model, "predict_proba") else preds
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    print(f"{name}: Acc={acc:.4f}, F1={f1:.4f}, AUC={auc:.4f}", flush=True)

"""
Test feature extraction and model benchmarking.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score

df = pd.read_csv("data/raw/synthetic_dna_variants.csv")

# Extract features
def extract_sample_features(row):
    seq = row["sequence"]
    seq_len = len(seq)
    count_a = seq.count("A") / seq_len
    count_c = seq.count("C") / seq_len
    count_g = seq.count("G") / seq_len
    count_t = seq.count("T") / seq_len
    gc = (seq.count("G") + seq.count("C")) / seq_len
    
    # 2-mer counts
    kmers_2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    kmer_feats = [seq.count(km) / max(1, seq_len - 1) for km in kmers_2]
    
    # Variant motif features
    ref = str(row.get("reference", ""))
    alt = str(row.get("alternate", ""))
    is_same = 1.0 if ref == alt else 0.0
    ref_len = len(ref)
    alt_len = len(alt)
    len_diff = alt_len - ref_len
    
    # Center motif
    center_idx = seq_len // 2
    center_window = seq[max(0, center_idx - 5): min(seq_len, center_idx + 6)]
    center_gc = (center_window.count("G") + center_window.count("C")) / max(1, len(center_window))
    
    # Triplet / codon motif around center
    triplet = seq[max(0, center_idx - 1): min(seq_len, center_idx + 2)]
    trip_gc = (triplet.count("G") + triplet.count("C")) / max(1, len(triplet))
    
    features = [
        count_a, count_c, count_g, count_t, gc,
        ref_len, alt_len, len_diff, is_same,
        center_gc, trip_gc
    ] + kmer_feats
    return features

X = np.array([extract_sample_features(r) for _, r in df.iterrows()])
y = df["label"].values

print(f"X shape: {X.shape}, y shape: {y.shape}", flush=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Test models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    "SVM": SVC(kernel="rbf", C=1.0, probability=True, random_state=42),
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

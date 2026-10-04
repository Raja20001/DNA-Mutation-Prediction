"""
Test pure sequence features with quantum phase trajectory and verify model accuracy.
"""
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

df = pd.read_csv("data/raw/synthetic_dna_variants.csv")

def extract_advanced_seq_features(seq):
    seq = str(seq).upper()
    seq_len = len(seq)
    if seq_len == 0:
        return [0.0] * 50
        
    f_a = seq.count("A") / seq_len
    f_c = seq.count("C") / seq_len
    f_g = seq.count("G") / seq_len
    f_t = seq.count("T") / seq_len
    gc = (seq.count("G") + seq.count("C")) / seq_len
    at = (seq.count("A") + seq.count("T")) / seq_len
    gc_skew = (seq.count("G") - seq.count("C")) / max(1, seq.count("G") + seq.count("C"))
    at_skew = (seq.count("A") - seq.count("T")) / max(1, seq.count("A") + seq.count("T"))
    
    # 2. Shannon entropy
    probs = [f for f in [f_a, f_c, f_g, f_t] if f > 0]
    shannon = -sum(p * math.log2(p) for p in probs)
    
    # 3. Gini-Simpson diversity
    diversity = 1.0 - sum(f**2 for f in [f_a, f_c, f_g, f_t])
    
    # 4. Dinucleotide frequencies (16)
    kmers_2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    k2 = [seq.count(km) / max(1, seq_len - 1) for km in kmers_2]
    
    # 5. Ti/Tv dinucleotide counts
    ti_count = seq.count("AG") + seq.count("GA") + seq.count("CT") + seq.count("TC")
    tv_count = (seq_len - 1) - ti_count
    ti_tv_ratio = ti_count / max(1, tv_count)
    
    # 6. CpG suppression index
    cpg_obs_exp = (seq.count("CG") * seq_len) / max(1, (seq.count("C") * seq.count("G")))
    
    # 7. Localized sliding-window entropy profile
    win_size = min(12, seq_len)
    step = 4
    local_ents = []
    for i in range(0, seq_len - win_size + 1, step):
        sub = seq[i : i + win_size]
        sub_len = len(sub)
        p_sub = [sub.count(b) / sub_len for b in "ACGT" if sub.count(b) > 0]
        local_ents.append(-sum(p * math.log2(p) for p in p_sub))
    ent_var = float(np.var(local_ents)) if len(local_ents) > 1 else 0.0
    ent_min = float(np.min(local_ents)) if len(local_ents) > 0 else 0.0
    ent_max = float(np.max(local_ents)) if len(local_ents) > 0 else 0.0
    
    # 8. Biological helical phase trajectory
    # Nucleotide phase angles: A=0, C=pi/2, G=pi, T=3pi/2
    angle_map = {"A": 0.0, "C": np.pi / 2, "G": np.pi, "T": 3 * np.pi / 2}
    angles = np.array([angle_map.get(b, 0.0) for b in seq])
    # Helical periodicity wave (10.5 bp standard B-DNA pitch)
    helical_freq = 2 * np.pi / 10.5
    indices = np.arange(seq_len)
    cos_wave = np.cos(angles + helical_freq * indices)
    sin_wave = np.sin(angles + helical_freq * indices)
    phase_r = np.mean(cos_wave)
    phase_i = np.mean(sin_wave)
    helical_coherence = float(np.sqrt(phase_r**2 + phase_i**2))
    helical_phase = float(np.arctan2(phase_i, phase_r))
    
    features = [
        f_a, f_c, f_g, f_t, gc, at, gc_skew, at_skew,
        shannon, diversity, ti_tv_ratio, cpg_obs_exp,
        ent_var, ent_min, ent_max, helical_coherence, helical_phase
    ] + k2
    return features

X = np.array([extract_advanced_seq_features(s) for s in df["sequence"]])
y = df["label"].values

print(f"X shape: {X.shape}", flush=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
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

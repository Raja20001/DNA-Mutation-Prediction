import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from src.preprocessing.synthetic_data import GENE_TEMPLATES
from src.preprocessing.cleaner import DNADataCleaner

HOTSPOT_SPECS = {
    "TP53": [
        {"type": "substitution", "ref_motif": "CGT", "alt_motif": "CAT", "desc": "c.818G>A (p.Arg273His) Hotspot"},
        {"type": "substitution", "ref_motif": "CGG", "alt_motif": "TGG", "desc": "c.742C>T (p.Arg248Trp) Hotspot"},
        {"type": "deletion", "ref_motif": "CCCCCT", "alt_motif": "CC", "desc": "c.112_115del Del-Frameshift"},
        {"type": "insertion", "ref_motif": "GAAAAC", "alt_motif": "GAAAAAAC", "desc": "c.154_155insAA Duplication"},
    ],
    "BRAF": [
        {"type": "substitution", "ref_motif": "GTG", "alt_motif": "GAG", "desc": "c.1799T>A (p.Val600Glu) Driver"},
        {"type": "substitution", "ref_motif": "GCT", "alt_motif": "GTT", "desc": "c.1789A>G (p.Ala598Val)"},
        {"type": "deletion", "ref_motif": "CTACAGTG", "alt_motif": "CTG", "desc": "c.1792_1796del In-Frame Del"},
        {"type": "insertion", "ref_motif": "TTTTGG", "alt_motif": "TTTTTTGG", "desc": "c.1780_1781insTT Insertion"},
    ],
    "KRAS": [
        {"type": "substitution", "ref_motif": "GGT", "alt_motif": "GAT", "desc": "c.35G>A (p.Gly12Asp) Driver"},
        {"type": "substitution", "ref_motif": "GGC", "alt_motif": "GTC", "desc": "c.38G>T (p.Gly13Val) Driver"},
        {"type": "insertion", "ref_motif": "GGTGGC", "alt_motif": "GGTGGTGGC", "desc": "c.34_36dup In-frame Dup"},
        {"type": "deletion", "ref_motif": "GTGGCG", "alt_motif": "GCG", "desc": "c.33_35del In-frame Del"},
    ],
    "EGFR": [
        {"type": "substitution", "ref_motif": "CTG", "alt_motif": "CGG", "desc": "c.2573T>G (p.Leu858Arg) Activating"},
        {"type": "deletion", "ref_motif": "GGAAGAGAAAGA", "alt_motif": "GGA", "desc": "c.2235_2246del Exon 19 Deletion"},
        {"type": "substitution", "ref_motif": "ACG", "alt_motif": "ATG", "desc": "c.2369C>T (p.Thr790Met) Resistance"},
        {"type": "insertion", "ref_motif": "GCCAA", "alt_motif": "GCCAACAA", "desc": "c.2210_2211insCAA Exon 20 Ins"},
    ],
    "BRCA1": [
        {"type": "deletion", "ref_motif": "AGTTCC", "alt_motif": "TTCC", "desc": "c.68_69delAG Founder Deletion"},
        {"type": "insertion", "ref_motif": "TTTTCA", "alt_motif": "TTTTCCA", "desc": "c.5266dupC (5382insC) Truncating"},
        {"type": "substitution", "ref_motif": "GAA", "alt_motif": "TAA", "desc": "c.181T>G Nonsense Termination"},
        {"type": "substitution", "ref_motif": "TGT", "alt_motif": "CGT", "desc": "c.190T>C (p.Cys64Arg) Deleterious"},
    ],
    "CFTR": [
        {"type": "deletion", "ref_motif": "ATCATCTTTGGT", "alt_motif": "ATCATTGGT", "desc": "c.1521_1523delCTT (deltaF508)"},
        {"type": "substitution", "ref_motif": "GGT", "alt_motif": "GAT", "desc": "c.1652G>A (p.Gly551Asp) Gating"},
        {"type": "substitution", "ref_motif": "CGT", "alt_motif": "TGT", "desc": "c.1000C>T (p.Arg334Trp)"},
        {"type": "insertion", "ref_motif": "TTTGGT", "alt_motif": "TTTTTTGGT", "desc": "c.1524_1525insTT Frameshift"},
    ],
}

def generate_anchored_dataset(n_samples=400, seed=42):
    genes = list(GENE_TEMPLATES.keys())
    records = []
    
    samples_per_gene = n_samples // len(genes) # 66
    win_len = 70
    
    for gene_name in genes:
        tmpl = GENE_TEMPLATES[gene_name]
        full_seq = tmpl["seq"]
        specs = HOTSPOT_SPECS[gene_name]
        
        n_gene_wt = samples_per_gene // 2 # 33
        n_gene_mut = samples_per_gene - n_gene_wt # 33
        max_start = max(0, len(full_seq) - win_len)
        
        # 1. Wildtypes: distinct even offsets (step by 1 or 2)
        wt_step = max(1, max_start // n_gene_wt)
        for i in range(n_gene_wt):
            start = min((i * wt_step), max_start)
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
            
        # 2. Mutants: anchored on hotspot motif with staggered placement
        for i in range(n_gene_mut):
            spec = specs[i % len(specs)]
            ref_motif = spec["ref_motif"]
            alt_motif = spec["alt_motif"]
            
            if ref_motif in full_seq:
                motif_idx = full_seq.index(ref_motif)
                # Stagger the motif position across the window (e.g. from pos 15 to pos 50)
                stagger = (i % 7) * 4 - 12 # -12, -8, -4, 0, 4, 8, 12
                start = max(0, min(max_start, motif_idx - win_len // 2 + stagger))
            else:
                start = (i * 2 + 1) % (max_start + 1)
                
            win = full_seq[start:start + win_len]
            if ref_motif in win:
                mut_win = win.replace(ref_motif, alt_motif, 1)
            else:
                c = len(win) // 2
                mut_win = win[:c] + alt_motif + win[c + len(ref_motif):]
                
            if len(mut_win) > win_len:
                mut_win = mut_win[:win_len]
            elif len(mut_win) < win_len:
                mut_win = mut_win + full_seq[start + len(mut_win) : start + win_len]
                
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
            
    return pd.DataFrame(records)

df = generate_anchored_dataset(400, seed=42)
cleaner = DNADataCleaner(remove_duplicates=True)
cleaned_df, audit = cleaner.clean_dataset(df)
print(f"Cleaned records: {len(cleaned_df)}, dropped duplicates: {audit['dropped_duplicates']}")

# Feature extraction matching pipeline: gc, at, kmer_2 (16), kmer_3 (64)
def extract_features(seq):
    seq = seq.upper()
    seq_len = len(seq)
    gc = (seq.count("G") + seq.count("C")) / seq_len
    at = (seq.count("A") + seq.count("T")) / seq_len
    
    kmers_2 = ["AA", "AC", "AG", "AT", "CA", "CC", "CG", "CT", "GA", "GC", "GG", "GT", "TA", "TC", "TG", "TT"]
    k2 = [seq.count(km) / (seq_len - 1) for km in kmers_2]
    
    from itertools import product
    kmers_3 = ["".join(p) for p in product("ACGT", repeat=3)]
    k3 = [seq.count(km) / (seq_len - 2) for km in kmers_3]
    return [gc, at] + k2 + k3

X = np.array([extract_features(s) for s in cleaned_df["sequence"]])
y = cleaned_df["label"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# Classical Models
rf = RandomForestClassifier(n_estimators=100, max_depth=7, random_state=42)
rf.fit(X_train_s, y_train)
rf_preds = rf.predict(X_test_s)
rf_acc = accuracy_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf.predict_proba(X_test_s)[:, 1])

gb = GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)
gb.fit(X_train_s, y_train)
gb_acc = accuracy_score(y_test, gb.predict(X_test_s))

print(f"Classical RF Accuracy: {rf_acc:.4f}, F1: {rf_f1:.4f}, AUC: {rf_auc:.4f}")
print(f"Classical GB Accuracy: {gb_acc:.4f}")

# HQ-CMFN
from src.qmfnet.models import HQCMFNClassifier

hqcmfn = HQCMFNClassifier(n_qubits=6, hidden_dim=64, epochs=140, lr=0.008, weight_decay=1e-4, random_state=42)
hqcmfn.fit(X_train_s, y_train)
hq_preds = hqcmfn.predict(X_test_s)
hq_acc = accuracy_score(y_test, hq_preds)
hq_f1 = f1_score(y_test, hq_preds)
hq_auc = roc_auc_score(y_test, hqcmfn.predict_proba(X_test_s)[:, 1])
print(f"HQ-CMFN (Quantum Hybrid) Accuracy: {hq_acc:.4f}, F1: {hq_f1:.4f}, AUC: {hq_auc:.4f}")

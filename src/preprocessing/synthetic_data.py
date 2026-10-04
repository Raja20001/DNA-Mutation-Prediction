"""
Synthetic DNA Variant Generator
Generates reproducible, in-silico synthetic DNA variant benchmark datasets
incorporating known human cancer/disease genes (*TP53*, *BRCA1*, *EGFR*, *BRAF*, *KRAS*, *CFTR*).
Every record is explicitly marked as synthetic to preserve scientific integrity.
"""
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Authentic gene reference sequences (exonic fragments for benchmarking)
GENE_TEMPLATES = {
    "TP53": {
        "chrom": "chr17",
        "start_pos": 7577120,
        "transcript": "ENST00000269305",
        "condition": "Li-Fraumeni Syndrome / Hereditary Cancer",
        # Exon 5-7 hotspot region fragment
        "seq": "ATGGAGGAGCCGCAGTCAGATCCTAGCGTCGAGCCCCCTCTGAGTCAGGAAACATTTTCAGACCTATGGAAACTACTTCCTGAAAACAACGTTCTGTCCCCCTTGCCGTCCCAAGCAATGGATGATTTGATGCTGTCCCCGGACGATATTGAACAATGGTTCAC",
    },
    "BRCA1": {
        "chrom": "chr17",
        "start_pos": 41276045,
        "transcript": "ENST00000357654",
        "condition": "Hereditary Breast and Ovarian Cancer",
        # Exon 11 fragment
        "seq": "AAGACCTGAAATGTACAGTGTTTTCTACACTAGGATTTTCTACACTTAGTTCCTATTTTGAATACTATTTTCAGGAGGACCTATTATTGTACTTGACAAAAGACAGCCAGCTGGATTACTACCAACATAGATTTT",
    },
    "BRAF": {
        "chrom": "chr7",
        "start_pos": 140453136,
        "transcript": "ENST00000288602",
        "condition": "Melanoma / Colorectal Cancer",
        # Exon 15 V600 hotspot fragment
        "seq": "CTACTGTTTTCCTTTACTTACTACACCTCAGATAATATATTTCTTCATGAAGACCTCACAGTAAAAATAGGTGATTTTGGTCTAGCTACAGTGAAATCTCGATGGAGTGGGTCCCATCAGTTTGAACAGTTG",
    },
    "EGFR": {
        "chrom": "chr7",
        "start_pos": 55242465,
        "transcript": "ENST00000275493",
        "condition": "Non-Small Cell Lung Carcinoma",
        # Exon 19 / 20 fragment
        "seq": "GATCACAGATTTTGGGCTGGCCAAACTGCTGGGTGCGGAAGAGAAAGAATACCATGCAGAAGGAGGCAAAGTGCCTATCAAGTGGATGGCATTGGAATCAATTTTACACAGAATCTATACCCACCAGAGTGAT",
    },
    "KRAS": {
        "chrom": "chr12",
        "start_pos": 25398284,
        "transcript": "ENST00000256078",
        "condition": "Colorectal and Pancreatic Carcinoma",
        # Codon 12/13 hotspot fragment
        "seq": "ACTTGTGGTAGTTGGAGCTGGTGGCGTAGGCAAGAGTGCCTTGACGATACAGCTAATTCAGAATCATTTTGTGGACGAATATGATCCAACAATAGAGGATTCCTACAGGAAGCAAGTAGTAATTGATGGA",
    },
    "CFTR": {
        "chrom": "chr7",
        "start_pos": 117199533,
        "transcript": "ENST00000003084",
        "condition": "Cystic Fibrosis",
        # Exon 10 (F508 region) fragment
        "seq": "TTTGTTTTCCTGGATTATGCCTGGCACCATTAAAGAAAATATCATCTTTGGTGTTTCCTATGATGAATATAGATACAGAAGCGTCATCAAAGCATGCCAACTAGAAGAGGACATCTCCAAGTTTGCAGAG",
    },
}

NUCLEOTIDES = ["A", "C", "G", "T"]


def introduce_mutation(
    ref_seq: str,
    mutation_type: str,
    rng: random.Random,
) -> Tuple[str, int, str, str, str]:
    """
    Introduce a specific type of mutation into a reference DNA sequence.

    Args:
        ref_seq: Original DNA sequence.
        mutation_type: 'substitution', 'insertion', 'deletion', or 'duplication'.
        rng: Seeded Random instance.

    Returns:
        Tuple[str, int, str, str, str]:
            (mutant_seq, 1_based_pos, ref_allele, alt_allele, hgvs_c_suffix)
    """
    seq_len = len(ref_seq)
    # Pick position with safe margins
    pos_0 = rng.randint(5, seq_len - 6)
    pos_1 = pos_0 + 1
    ref_char = ref_seq[pos_0]

    if mutation_type == "substitution":
        alt_candidates = [n for n in NUCLEOTIDES if n != ref_char]
        alt_char = rng.choice(alt_candidates)
        mutant_seq = ref_seq[:pos_0] + alt_char + ref_seq[pos_0 + 1:]
        hgvs_suffix = f"c.{pos_1}{ref_char}>{alt_char}"
        return mutant_seq, pos_1, ref_char, alt_char, hgvs_suffix

    elif mutation_type == "insertion":
        ins_length = rng.randint(1, 3)
        inserted_bases = "".join(rng.choices(NUCLEOTIDES, k=ins_length))
        mutant_seq = ref_seq[:pos_0] + inserted_bases + ref_seq[pos_0:]
        alt_allele = ref_char + inserted_bases
        hgvs_suffix = f"c.{pos_1}_{pos_1+1}ins{inserted_bases}"
        return mutant_seq, pos_1, ref_char, alt_allele, hgvs_suffix

    elif mutation_type == "deletion":
        del_length = rng.randint(1, 3)
        end_0 = min(pos_0 + del_length, seq_len - 2)
        deleted_bases = ref_seq[pos_0:end_0]
        mutant_seq = ref_seq[:pos_0] + ref_seq[end_0:]
        hgvs_suffix = f"c.{pos_1}_{pos_1 + len(deleted_bases) - 1}del"
        return mutant_seq, pos_1, deleted_bases, "-", hgvs_suffix

    elif mutation_type == "duplication":
        dup_len = rng.randint(1, 3)
        dup_seq = ref_seq[pos_0:pos_0 + dup_len]
        mutant_seq = ref_seq[:pos_0 + dup_len] + dup_seq + ref_seq[pos_0 + dup_len:]
        hgvs_suffix = f"c.{pos_1}_{pos_1 + dup_len - 1}dup"
        return mutant_seq, pos_1, dup_seq, dup_seq * 2, hgvs_suffix

    else:
        raise ValueError(f"Unknown mutation type: {mutation_type}")


def generate_synthetic_dataset(
    n_samples: int = 400,
    seed: int = 42,
    mutation_ratio: float = 0.5,
    num_samples: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a realistic in-silico benchmark dataset with balanced wildtypes and mutations.

    Args:
        n_samples: Total number of sequence records to generate.
        seed: Random seed for exact reproducibility.
        mutation_ratio: Fraction of samples that are mutants (label = 1).
        num_samples: Alias for n_samples.
        random_seed: Alias for seed.

    Returns:
        pd.DataFrame: Formatted DNA variant benchmark dataset.
    """
    if num_samples is not None:
        n_samples = num_samples
    if random_seed is not None:
        seed = random_seed
    HOTSPOT_SPECS: Dict[str, List[Dict[str, str]]] = {
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

    rng = random.Random(seed)
    genes = list(GENE_TEMPLATES.keys())
    records: List[Dict] = []
    seen_sequences: set = set()
    
    samples_per_gene = n_samples // len(genes)
    for gene_name in genes:
        tmpl = GENE_TEMPLATES[gene_name]
        full_seq = tmpl["seq"]
        specs = HOTSPOT_SPECS[gene_name]
        
        n_gene_wt = samples_per_gene // 2
        n_gene_mut = samples_per_gene - n_gene_wt
        
        win_len = 70
        max_start = max(0, len(full_seq) - win_len)
        
        # 1. Generate Wildtypes (unique sampled windows across canonical exon)
        wt_attempts = 0
        while len([r for r in records if r["gene"] == gene_name and r["label"] == 0]) < n_gene_wt and wt_attempts < 200:
            wt_attempts += 1
            start = rng.randint(0, max_start) if max_start > 0 else 0
            win = full_seq[start:start + win_len]
            if len(win) == win_len and win not in seen_sequences:
                seen_sequences.add(win)
                pos = tmpl["start_pos"] + start + win_len // 2
                records.append({
                    "variant_id": f"SYN_WT_{gene_name}_{len([r for r in records if r['gene'] == gene_name and r['label'] == 0]) + 1:04d}",
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

        # 2. Generate Mutants (authentic clinical hotspot alterations with varied window context)
        mut_attempts = 0
        while len([r for r in records if r["gene"] == gene_name and r["label"] == 1]) < n_gene_mut and mut_attempts < 200:
            mut_attempts += 1
            idx_in_gene = len([r for r in records if r["gene"] == gene_name and r["label"] == 1])
            spec = specs[idx_in_gene % len(specs)]
            ref_motif = spec["ref_motif"]
            alt_motif = spec["alt_motif"]

            start = rng.randint(0, max_start) if max_start > 0 else 0
            win = full_seq[start:start + win_len]
            if ref_motif in win:
                mut_win = win.replace(ref_motif, alt_motif, 1)
            else:
                c = rng.randint(15, win_len - 15)
                mut_win = win[:c] + alt_motif + win[c + len(alt_motif):]

            if len(mut_win) > win_len:
                mut_win = mut_win[:win_len]
            elif len(mut_win) < win_len:
                mut_win = mut_win + "A" * (win_len - len(mut_win))

            if mut_win not in seen_sequences:
                seen_sequences.add(mut_win)
                pos = tmpl["start_pos"] + start + win_len // 2
                records.append({
                    "variant_id": f"SYN_MUT_{gene_name}_{idx_in_gene + 1:04d}",
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

    # Shuffle records reproducibly
    rng.shuffle(records)
    return pd.DataFrame(records)


def generate_edge_case_dataset(seed: int = 42) -> pd.DataFrame:
    """
    Generate an edge-case dataset containing deliberate quality issues:
    - IUPAC ambiguous codes ('N', 'R', 'Y')
    - Lowercase sequences
    - Whitespace and newlines
    - Invalid non-DNA characters ('Z', 'X', '9')
    - Empty and NaN sequences
    - Duplicate entries
    - Extremely short or long sequences

    Used for comprehensive unit and integration testing of the validator and cleaner.
    """
    rng = random.Random(seed)
    edge_records = [
        # Normal baseline
        {"sequence": "ATGCGTACGTTAGC", "label": 0, "gene": "TEST1", "note": "valid_standard"},
        # Lowercase
        {"sequence": "atgcgtacgttagc", "label": 1, "gene": "TEST1", "note": "lowercase"},
        # Embedded whitespace and newlines
        {"sequence": "ATG C\tGTA\nCGTT AGC", "label": 0, "gene": "TEST2", "note": "whitespace_newlines"},
        # IUPAC ambiguous codes
        {"sequence": "ATGCRTAYGTNAGC", "label": 1, "gene": "TEST2", "note": "iupac_ambiguity"},
        # Invalid non-DNA characters
        {"sequence": "ATGCZ9XTAGC", "label": 0, "gene": "TEST3", "note": "invalid_characters"},
        # Too short (< 10 bp)
        {"sequence": "ATGC", "label": 1, "gene": "TEST3", "note": "too_short"},
        # Missing sequence (None)
        {"sequence": None, "label": 0, "gene": "TEST4", "note": "none_sequence"},
        # Missing label (None)
        {"sequence": "ATGCGTAGCTAGCTAGCTA", "label": None, "gene": "TEST4", "note": "none_label"},
        # Duplicate pair
        {"sequence": "GGCCAATTGGCCAATTGGCC", "label": 1, "gene": "TEST5", "note": "duplicate_a"},
        {"sequence": "GGCCAATTGGCCAATTGGCC", "label": 1, "gene": "TEST5", "note": "duplicate_b"},
    ]
    return pd.DataFrame(edge_records)

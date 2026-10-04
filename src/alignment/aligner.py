"""
Robust Sequence Alignment Engine & Mutation Visualizer
Detects:
- SNV (Single Nucleotide Variant)
- MNV (Multi-Nucleotide Variant)
- Insertion
- Deletion
- Duplication
- Complex DELINS
Generates visual alignment representations with match bars, mismatch markers, and HTML rendering.
"""
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union


@dataclass
class EditOperation:
    """Represents an atomic edit event between reference and alternate sequences."""
    op_type: str                   # 'match', 'substitution', 'insertion', 'deletion', 'duplication', 'delins'
    ref_pos: Optional[int] = None   # 1-based coordinate in reference (None for insertions)
    alt_pos: Optional[int] = None   # 1-based coordinate in alternate (None for deletions)
    ref_seq: str = ""
    alt_seq: str = ""
    description: str = ""


@dataclass
class AlignmentResult:
    """Standardized output from pairwise genomic alignment."""
    aligned_reference: str
    aligned_alternate: str
    alignment_match_string: str
    edit_operations: List[Dict[str, Any]]
    mutation_positions: List[int]
    confidence: float
    mutation_type: str             # 'WILDTYPE', 'SNV', 'MNV', 'INSERTION', 'DELETION', 'DUPLICATION', 'DELINS', 'UNKNOWN'
    mutation_subtype: str          # 'Transition', 'Transversion', 'In-frame', 'Frameshift', 'Identical', 'Complex'
    primary_change: str            # e.g., 'C>T', 'delTGG', 'insA', 'dupC'
    reference_base: str            # e.g., 'C'
    alternate_base: str            # e.g., 'T'
    primary_position: Optional[int] = None
    flanking_context: str = ""
    visual_ascii: str = ""
    visual_html: str = ""

    @property
    def edit_distance(self) -> int:
        return len(self.edit_operations)

    @property
    def identity_percent(self) -> float:
        if not self.alignment_match_string:
            return 0.0
        matches = self.alignment_match_string.count("|")
        return round((matches / len(self.alignment_match_string)) * 100.0, 2)

    @property
    def score(self) -> float:
        return float(self.confidence * 100.0)

    @property
    def visual_alignment(self) -> str:
        return self.visual_ascii

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)



def needleman_wunsch_align(
    seq1: str,
    seq2: str,
    match_score: int = 2,
    mismatch_score: int = -1,
    gap_penalty: int = -2,
) -> Tuple[str, str]:
    """
    Standard Needleman-Wunsch global pairwise alignment algorithm.
    Guarantees mathematically optimal sequence alignment under specified scoring parameters.
    """
    n = len(seq1)
    m = len(seq2)

    # Initialize DP matrix
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i * gap_penalty
    for j in range(m + 1):
        dp[0][j] = j * gap_penalty

    # Fill DP matrix
    for i in range(1, n + 1):
        b1 = seq1[i - 1]
        for j in range(1, m + 1):
            b2 = seq2[j - 1]
            score_diag = dp[i - 1][j - 1] + (match_score if b1 == b2 else mismatch_score)
            score_up = dp[i - 1][j] + gap_penalty
            score_left = dp[i][j - 1] + gap_penalty
            dp[i][j] = max(score_diag, score_up, score_left)

    # Traceback
    aligned1 = []
    aligned2 = []
    i, j = n, m

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            b1 = seq1[i - 1]
            b2 = seq2[j - 1]
            expected_diag = dp[i - 1][j - 1] + (match_score if b1 == b2 else mismatch_score)
            if dp[i][j] == expected_diag:
                aligned1.append(b1)
                aligned2.append(b2)
                i -= 1
                j -= 1
                continue

        if i > 0 and (j == 0 or dp[i][j] == dp[i - 1][j] + gap_penalty):
            aligned1.append(seq1[i - 1])
            aligned2.append("-")
            i -= 1
        else:
            aligned1.append("-")
            aligned2.append(seq2[j - 1])
            j -= 1

    aligned_ref = "".join(reversed(aligned1))
    aligned_alt = "".join(reversed(aligned2))
    return aligned_ref, aligned_alt


def is_transition(ref_base: str, alt_base: str) -> bool:
    """Return True if SNV is a transition (A<->G or C<->T), False if transversion."""
    purines = {"A", "G"}
    pyrimidines = {"C", "T"}
    r = ref_base.upper()
    a = alt_base.upper()
    return (r in purines and a in purines) or (r in pyrimidines and a in pyrimidines)


def align_sequences(
    reference: Optional[str],
    alternate: Optional[str],
    genomic_position_offset: Optional[int] = None,
    window_context: int = 5,
) -> AlignmentResult:
    """
    Robust sequence alignment and variant classification engine.
    Detects SNV, MNV, Insertion, Deletion, Duplication, DELINS, and Wildtype.
    """
    if not reference or not alternate:
        return AlignmentResult(
            aligned_reference="",
            aligned_alternate="",
            alignment_match_string="",
            edit_operations=[],
            mutation_positions=[],
            confidence=0.0,
            mutation_type="UNKNOWN",
            mutation_subtype="Missing Sequence Information",
            primary_change="N/A",
            reference_base="N/A",
            alternate_base="N/A",
            primary_position=None,
            flanking_context="N/A",
            visual_ascii="Alignment unavailable: Missing sequence inputs.",
            visual_html="<div class='text-muted'>Alignment unavailable</div>",
        )

    ref = "".join(str(reference).split()).upper()
    alt = "".join(str(alternate).split()).upper()

    # Identical sequence fast path (Wildtype)
    if ref == alt:
        match_str = "|" * len(ref)
        ascii_disp = f"REF 5' {ref} 3'\n       {match_str}\nALT 5' {alt} 3'"
        return AlignmentResult(
            aligned_reference=ref,
            aligned_alternate=alt,
            alignment_match_string=match_str,
            edit_operations=[{"op_type": "match", "ref_pos": 1, "alt_pos": 1, "ref_seq": ref, "alt_seq": alt}],
            mutation_positions=[],
            confidence=1.0,
            mutation_type="WILDTYPE",
            mutation_subtype="Identical Reference Match",
            primary_change="None (Identical)",
            reference_base="-",
            alternate_base="-",
            primary_position=None,
            flanking_context=ref[: min(len(ref), window_context * 2 + 1)],
            visual_ascii=ascii_disp,
            visual_html=f"<div style='font-family: monospace;'><strong>REF:</strong> {ref}<br><strong>&nbsp;&nbsp;&nbsp;&nbsp;</strong> {match_str}<br><strong>ALT:</strong> {alt}</div>",
        )

    # Global Needleman-Wunsch alignment
    aln_ref, aln_alt = needleman_wunsch_align(ref, alt)

    # Build match string and parse edit operations
    match_chars = []
    edit_ops = []
    mut_positions = []

    curr_ref_pos = 1
    curr_alt_pos = 1

    diff_blocks = []
    in_diff = False
    diff_start_ref = None
    diff_start_alt = None
    diff_ref_buf = []
    diff_alt_buf = []

    for idx, (b_ref, b_alt) in enumerate(zip(aln_ref, aln_alt)):
        if b_ref == b_alt:
            match_chars.append("|")
            if in_diff:
                # Flush active difference block
                diff_blocks.append({
                    "ref_pos": diff_start_ref,
                    "alt_pos": diff_start_alt,
                    "ref_seq": "".join(diff_ref_buf).replace("-", ""),
                    "alt_seq": "".join(diff_alt_buf).replace("-", ""),
                })
                in_diff = False
                diff_ref_buf = []
                diff_alt_buf = []
        else:
            if not in_diff:
                in_diff = True
                diff_start_ref = curr_ref_pos if b_ref != "-" else curr_ref_pos
                diff_start_alt = curr_alt_pos if b_alt != "-" else curr_alt_pos
            diff_ref_buf.append(b_ref)
            diff_alt_buf.append(b_alt)

            if b_ref == "-":
                match_chars.append("+")  # Insertion
            elif b_alt == "-":
                match_chars.append("-")  # Deletion
            else:
                match_chars.append("*")  # Mismatch / Substitution
            
            mut_positions.append(curr_ref_pos)

        if b_ref != "-":
            curr_ref_pos += 1
        if b_alt != "-":
            curr_alt_pos += 1

    if in_diff:
        diff_blocks.append({
            "ref_pos": diff_start_ref,
            "alt_pos": diff_start_alt,
            "ref_seq": "".join(diff_ref_buf).replace("-", ""),
            "alt_seq": "".join(diff_alt_buf).replace("-", ""),
        })

    match_str = "".join(match_chars)

    # Classify mutation category based on difference blocks
    mutation_type = "UNKNOWN"
    mutation_subtype = "Unknown"
    primary_change = "N/A"
    primary_ref = "-"
    primary_alt = "-"
    first_pos = 1

    if len(diff_blocks) == 1:
        blk = diff_blocks[0]
        r_diff = blk["ref_seq"]
        a_diff = blk["alt_seq"]
        first_pos = blk["ref_pos"]

        primary_ref = r_diff or "-"
        primary_alt = a_diff or "-"

        if len(r_diff) == 1 and len(a_diff) == 1:
            mutation_type = "SNV"
            subtype = "Transition" if is_transition(r_diff, a_diff) else "Transversion"
            mutation_subtype = subtype
            primary_change = f"{r_diff}>{a_diff}"
        elif len(r_diff) > 1 and len(a_diff) == len(r_diff):
            mutation_type = "MNV"
            mutation_subtype = f"{len(r_diff)}-bp Multi-Nucleotide Variant"
            primary_change = f"{r_diff}>{a_diff}"
        elif len(r_diff) == 0 and len(a_diff) > 0:
            # Check duplication: does inserted sequence duplicate immediately preceding reference bases?
            pos_0 = (first_pos - 1) if first_pos else 0
            prec_ref = ref[max(0, pos_0 - len(a_diff)) : pos_0]
            if prec_ref == a_diff:
                mutation_type = "DUPLICATION"
                mutation_subtype = "Tandem Duplication"
                primary_change = f"dup{a_diff}"
            else:
                mutation_type = "INSERTION"
                mutation_subtype = f"{len(a_diff)}-bp Insertion"
                primary_change = f"ins{a_diff}"
        elif len(r_diff) > 0 and len(a_diff) == 0:
            mutation_type = "DELETION"
            mutation_subtype = f"{len(r_diff)}-bp Deletion"
            primary_change = f"del{r_diff}"
        else:
            mutation_type = "DELINS"
            mutation_subtype = f"Complex Delins ({len(r_diff)}bp replaced by {len(a_diff)}bp)"
            primary_change = f"del{r_diff}ins{a_diff}"
    elif len(diff_blocks) > 1:
        mutation_type = "DELINS"
        mutation_subtype = f"Multiple Discontinuous Variations ({len(diff_blocks)} events)"
        primary_change = f"{diff_blocks[0]['ref_seq']}>{diff_blocks[0]['alt_seq']} (complex)"
        first_pos = diff_blocks[0]["ref_pos"]
        primary_ref = diff_blocks[0]["ref_seq"] or "-"
        primary_alt = diff_blocks[0]["alt_seq"] or "-"

    # Genomic coordinate calculation
    final_pos = (genomic_position_offset + first_pos - 1) if (genomic_position_offset and first_pos) else first_pos

    # Flanking context
    start_ctx = max(0, (first_pos or 1) - 1 - window_context)
    end_ctx = min(len(ref), (first_pos or 1) + max(1, len(primary_ref.replace("-", ""))) + window_context)
    flank_left = ref[start_ctx : (first_pos or 1) - 1]
    flank_right = ref[(first_pos or 1) + len(primary_ref.replace("-", "")) - 1 : end_ctx]
    context_str = f"{flank_left}[{primary_ref}>{primary_alt}]{flank_right}"

    # Visual ASCII representation
    marker_line = "".join("^" if c in ("*", "+", "-") else " " for c in match_chars)
    visual_ascii = (
        f"REFERENCE 5' {aln_ref} 3'\n"
        f"             {match_str}\n"
        f"SAMPLE    5' {aln_alt} 3'\n"
        f"             {marker_line}\n"
        f"             Position: {final_pos} | Change: {primary_change} ({mutation_type} - {mutation_subtype})"
    )

    # Visual HTML representation
    ref_spans = []
    alt_spans = []
    for r_ch, a_ch in zip(aln_ref, aln_alt):
        if r_ch == a_ch:
            ref_spans.append(f"<span style='color: #94A3B8;'>{r_ch}</span>")
            alt_spans.append(f"<span style='color: #94A3B8;'>{a_ch}</span>")
        elif r_ch == "-":
            ref_spans.append("<span style='color: #64748B;'>-</span>")
            alt_spans.append(f"<span style='background: #10B981; color: #FFF; font-weight:bold; padding:0 2px; border-radius:2px;'>{a_ch}</span>")
        elif a_ch == "-":
            ref_spans.append(f"<span style='background: #EF4444; color: #FFF; font-weight:bold; padding:0 2px; border-radius:2px;'>{r_ch}</span>")
            alt_spans.append("<span style='color: #64748B;'>-</span>")
        else:
            ref_spans.append(f"<span style='background: #F59E0B; color: #000; font-weight:bold; padding:0 2px; border-radius:2px;'>{r_ch}</span>")
            alt_spans.append(f"<span style='background: #38BDF8; color: #000; font-weight:bold; padding:0 2px; border-radius:2px;'>{a_ch}</span>")

    visual_html = (
        f"<div style='font-family: monospace; font-size: 0.95rem; line-height: 1.6; background: #0F172A; padding: 12px; border-radius: 8px; border: 1px solid #1E293B;'>"
        f"<div><span style='color: #64748B; width: 60px; display:inline-block;'>REF:</span> {''.join(ref_spans)}</div>"
        f"<div><span style='color: #64748B; width: 60px; display:inline-block;'>&nbsp;</span> {match_str.replace(' ', '&nbsp;')}</div>"
        f"<div><span style='color: #64748B; width: 60px; display:inline-block;'>ALT:</span> {''.join(alt_spans)}</div>"
        f"<div style='margin-top: 8px; color: #38BDF8; font-size: 0.85rem;'>Mutation: <strong>{primary_change}</strong> at locus <strong>{final_pos}</strong> ({mutation_type}: {mutation_subtype})</div>"
        f"</div>"
    )

    return AlignmentResult(
        aligned_reference=aln_ref,
        aligned_alternate=aln_alt,
        alignment_match_string=match_str,
        edit_operations=diff_blocks,
        mutation_positions=[(genomic_position_offset + p - 1) if genomic_position_offset else p for p in mut_positions],
        confidence=0.99,
        mutation_type=mutation_type,
        mutation_subtype=mutation_subtype,
        primary_change=primary_change,
        reference_base=primary_ref,
        alternate_base=primary_alt,
        primary_position=final_pos,
        flanking_context=context_str,
        visual_ascii=visual_ascii,
        visual_html=visual_html,
    )

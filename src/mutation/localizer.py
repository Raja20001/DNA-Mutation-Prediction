"""
Mutation Localization Module
Pinpoints the exact nucleotide coordinate, reference base, alternate base,
mutation event (e.g., C>T, insA, delGT), and local sequence context window.

CRITICAL RESEARCH RULE:
Never invent a position. If sequences cannot be reliably localized or aligned,
explicitly state: "Localization unavailable from supplied sequence information."
"""
from typing import Any, Dict, Optional, Tuple


def localize_mutation(
    reference_seq: Optional[str],
    alternate_seq: Optional[str],
    window_size: int = 5,
    genomic_position: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Localize mutation between reference and alternate sequences.

    Args:
        reference_seq: Reference DNA sequence string.
        alternate_seq: Alternate/mutated DNA sequence string.
        window_size: Flanking base pairs on each side for local context.
        genomic_position: Optional chromosomal genomic position offset.

    Returns:
        Dict[str, Any] containing:
            - localized: bool
            - position: 1-based local index or genomic coordinate, or None
            - reference_base: str
            - alternate_base: str
            - change: str (e.g., 'C>T', 'insA', 'delTGG')
            - mutation_type: 'substitution', 'insertion', 'deletion', 'duplication', or 'wildtype'
            - sequence_context: flanking context string
            - message: status explanation
    """
    unavailable_msg = "Localization unavailable from supplied sequence information."

    if not reference_seq or not alternate_seq:
        return {
            "localized": False,
            "position": None,
            "reference_base": "N/A",
            "alternate_base": "N/A",
            "change": "N/A",
            "mutation_type": "unknown",
            "sequence_context": "N/A",
            "message": unavailable_msg,
        }

    ref = str(reference_seq).strip().upper()
    alt = str(alternate_seq).strip().upper()

    if ref == alt:
        return {
            "localized": True,
            "position": None,
            "reference_base": "N/A",
            "alternate_base": "N/A",
            "change": "None (Identical)",
            "mutation_type": "wildtype",
            "sequence_context": ref[: min(len(ref), window_size * 2 + 1)],
            "message": "Sequences are identical (Wildtype).",
        }

    # Find first mismatch from 5' end
    prefix_len = 0
    min_len = min(len(ref), len(alt))
    while prefix_len < min_len and ref[prefix_len] == alt[prefix_len]:
        prefix_len += 1

    # Find matching suffix from 3' end
    suffix_ref = len(ref) - 1
    suffix_alt = len(alt) - 1
    while suffix_ref >= prefix_len and suffix_alt >= prefix_len and ref[suffix_ref] == alt[suffix_alt]:
        suffix_ref -= 1
        suffix_alt -= 1

    diff_ref = ref[prefix_len : suffix_ref + 1]
    diff_alt = alt[prefix_len : suffix_alt + 1]

    # Calculate 1-based relative position
    local_pos = prefix_len + 1
    actual_pos = (genomic_position + prefix_len) if genomic_position is not None else local_pos

    # Determine mutation type and change notation
    if len(diff_ref) == len(diff_alt) == 1:
        mutation_type = "substitution"
        change = f"{diff_ref}>{diff_alt}"
    elif len(diff_ref) == 0 and len(diff_alt) > 0:
        # Check if insertion is a duplication of preceding bases
        dup_window = ref[max(0, prefix_len - len(diff_alt)) : prefix_len]
        if dup_window == diff_alt:
            mutation_type = "duplication"
            change = f"dup{diff_alt}"
        else:
            mutation_type = "insertion"
            change = f"ins{diff_alt}"
    elif len(diff_ref) > 0 and len(diff_alt) == 0:
        mutation_type = "deletion"
        change = f"del{diff_ref}"
    else:
        mutation_type = "indel_complex"
        change = f"{diff_ref or '-'}>{diff_alt or '-'}"

    # Extract sequence context window around the mutation
    ctx_start = max(0, prefix_len - window_size)
    ctx_end = min(len(ref), prefix_len + max(1, len(diff_ref)) + window_size)
    flank_left = ref[ctx_start:prefix_len]
    flank_right = ref[prefix_len + len(diff_ref) : ctx_end]
    context_display = f"{flank_left}[{diff_ref or '-'}>{diff_alt or '-'}]{flank_right}"

    return {
        "localized": True,
        "position": actual_pos,
        "local_position": local_pos,
        "reference_base": diff_ref or "-",
        "alternate_base": diff_alt or "-",
        "change": change,
        "mutation_type": mutation_type,
        "sequence_context": context_display,
        "context_flank_5p": flank_left,
        "context_flank_3p": flank_right,
        "message": f"Successfully localized mutation at position {actual_pos}.",
    }

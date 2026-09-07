"""Text-decoding accuracy metrics: character error rate and exact match.

Small, dependency-free implementation (no `jiwer`/`editdistance` package) --
straightforward Levenshtein edit distance is all that's needed for
alphanumeric CW message strings.
"""
from __future__ import annotations


def _levenshtein_distance(a: str, b: str) -> int:
    """Standard O(len(a)*len(b)) dynamic-programming edit distance."""
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    previous_row = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current_row = [i] + [0] * len(b)
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current_row[j] = min(
                previous_row[j] + 1,       # deletion
                current_row[j - 1] + 1,    # insertion
                previous_row[j - 1] + cost,  # substitution
            )
        previous_row = current_row
    return previous_row[-1]


def character_error_rate(pred: str, ref: str) -> float:
    """Levenshtein distance between `pred` and `ref`, normalized by `len(ref)`.

    Can exceed 1.0 if `pred` is much longer than `ref` (more insertions than
    reference characters). A reference of length 0 returns 0.0 if `pred` is
    also empty, else 1.0 per inserted character relative to an assumed
    single-character reference (avoids a division by zero).
    """
    if len(ref) == 0:
        return 0.0 if len(pred) == 0 else float(len(pred))
    return _levenshtein_distance(pred, ref) / len(ref)


def exact_match(pred: str, ref: str) -> bool:
    """Whether `pred` exactly equals `ref`."""
    return pred == ref


def mean_character_error_rate(preds: list[str], refs: list[str]) -> float:
    """Mean CER over a list of (pred, ref) pairs."""
    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must be the same length, got {len(preds)} vs {len(refs)}")
    if not preds:
        return 0.0
    return sum(character_error_rate(p, r) for p, r in zip(preds, refs)) / len(preds)


def exact_match_rate(preds: list[str], refs: list[str]) -> float:
    """Fraction of (pred, ref) pairs that are an exact match."""
    if len(preds) != len(refs):
        raise ValueError(f"preds and refs must be the same length, got {len(preds)} vs {len(refs)}")
    if not preds:
        return 0.0
    return sum(exact_match(p, r) for p, r in zip(preds, refs)) / len(preds)

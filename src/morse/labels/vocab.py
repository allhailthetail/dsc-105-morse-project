"""Character vocabulary and text <-> integer-sequence encoding for CTC training.

CTC loss requires an extra "blank" symbol beyond the actual character set, so
the vocabulary here is the 36 alphanumeric characters (matching
`morse.morse_code.ALPHABET`) plus one reserved blank index.
"""
from __future__ import annotations

import numpy as np

from morse.morse_code import ALPHABET

# Blank is placed after the real characters so index 0..len(ALPHABET)-1 always
# map directly to real characters -- convenient for anything that ignores CTC
# blanks by simply masking out the last index.
BLANK_INDEX = len(ALPHABET)
VOCAB_SIZE = len(ALPHABET) + 1

_CHAR_TO_ID = {ch: i for i, ch in enumerate(ALPHABET)}
_ID_TO_CHAR = {i: ch for ch, i in _CHAR_TO_ID.items()}


def encode(text: str) -> np.ndarray:
    """Text (A-Z/0-9 only) -> int32 array of character ids (no blanks inserted)."""
    text = text.upper()
    try:
        return np.array([_CHAR_TO_ID[ch] for ch in text], dtype=np.int32)
    except KeyError as exc:
        raise ValueError(f"unsupported character {exc.args[0]!r} in {text!r}") from exc


def decode(ids: np.ndarray | list[int]) -> str:
    """Int character ids (no blanks) -> text. Inverse of `encode`."""
    return "".join(_ID_TO_CHAR[int(i)] for i in ids)


def ctc_collapse(ids: np.ndarray | list[int]) -> str:
    """Collapse a raw per-frame CTC output (with blanks and repeats) into text.

    Standard CTC decoding: merge consecutive repeated symbols, then drop blanks.
    Useful for turning a model's argmax-per-frame output into a final string.
    """
    collapsed: list[int] = []
    previous: int | None = None
    for i in ids:
        i = int(i)
        if i != previous:
            collapsed.append(i)
        previous = i
    return decode([i for i in collapsed if i != BLANK_INDEX])

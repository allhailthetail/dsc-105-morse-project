"""The ITU Morse code table and text <-> dit/dah encode/decode helpers.

Covers A-Z and 0-9, matching the alphanumeric format of the archived real
dataset (`data/real/labels.csv`). Elements are represented as strings of
"." (dit) and "-" (dah); characters within a message are implicitly
separated by an inter-character gap and words by a word gap when rendered
to timing (see `morse.timing`) -- this module only handles the symbolic
dit/dah mapping, not durations.
"""
from __future__ import annotations

MORSE_TABLE: dict[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".",
    "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
    "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---",
    "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--",
    "Z": "--..",
    "0": "-----", "1": ".----", "2": "..---", "3": "...--",
    "4": "....-", "5": ".....", "6": "-....", "7": "--...",
    "8": "---..", "9": "----.",
}

# Inverse mapping, built once at import time.
_REVERSE_TABLE: dict[str, str] = {code: char for char, code in MORSE_TABLE.items()}

ALPHABET: str = "".join(MORSE_TABLE.keys())


def text_to_morse(text: str) -> list[str]:
    """Convert a text string to a list of dit/dah code strings, one per character.

    Raises ValueError on any character outside A-Z/0-9 (after uppercasing).
    Whitespace is not supported here -- callers wanting word breaks should
    split on spaces before calling and handle word-gap timing separately.
    """
    text = text.upper()
    try:
        return [MORSE_TABLE[ch] for ch in text]
    except KeyError as exc:
        raise ValueError(f"unsupported character {exc.args[0]!r} in {text!r}") from exc


def morse_to_text(codes: list[str]) -> str:
    """Convert a list of dit/dah code strings back to text. Inverse of text_to_morse."""
    try:
        return "".join(_REVERSE_TABLE[code] for code in codes)
    except KeyError as exc:
        raise ValueError(f"unrecognized Morse code {exc.args[0]!r}") from exc

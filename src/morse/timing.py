"""PARIS-standard CW timing.

The PARIS standard defines Morse code speed (WPM) by treating the word
"PARIS" (including its trailing word-space) as exactly 50 time units (tau):

    tau = 6 / (5 * wpm)     # seconds per time-unit, derived from 50*tau = 60/wpm

Element durations, in units of tau:
    dot                  = 1 * tau
    dash                 = 3 * tau
    intra-character gap  = 1 * tau   (between dits/dahs of the same character)
    inter-character gap  = 3 * tau   (between characters of the same word)
    word gap             = 7 * tau   (between words)

See legacy/archive/docs/PARIS-standard.tex for the original derivation.
"""
from __future__ import annotations

from dataclasses import dataclass

DOT_UNITS = 1
DASH_UNITS = 3
INTRA_CHAR_GAP_UNITS = 1
INTER_CHAR_GAP_UNITS = 3
WORD_GAP_UNITS = 7


def tau_seconds(wpm: float) -> float:
    """Seconds per PARIS time-unit at the given words-per-minute speed."""
    if wpm <= 0:
        raise ValueError(f"wpm must be positive, got {wpm}")
    return 6.0 / (5.0 * wpm)


def approx_wpm(num_chars: int, duration_s: float) -> float:
    """Approximate WPM from a fixed-length message's total duration.

    Matches the heuristic used in the archived R analysis (main.org): for a
    message of a known, fixed character count, WPM ~= 12 * num_chars / duration.
    This is only a rough estimate (it ignores actual inter-word/inter-character
    spacing) -- prefer `tau_seconds`/`ElementTiming` for anything generated
    synthetically, where the true WPM is known exactly.
    """
    if duration_s <= 0:
        raise ValueError(f"duration_s must be positive, got {duration_s}")
    return round(12.0 * num_chars / duration_s, 2)


@dataclass(frozen=True)
class ElementTiming:
    """Durations (seconds) for the five PARIS timing elements at a given WPM."""

    wpm: float
    dot: float
    dash: float
    intra_char_gap: float
    inter_char_gap: float
    word_gap: float

    @classmethod
    def at(cls, wpm: float) -> "ElementTiming":
        tau = tau_seconds(wpm)
        return cls(
            wpm=wpm,
            dot=DOT_UNITS * tau,
            dash=DASH_UNITS * tau,
            intra_char_gap=INTRA_CHAR_GAP_UNITS * tau,
            inter_char_gap=INTER_CHAR_GAP_UNITS * tau,
            word_gap=WORD_GAP_UNITS * tau,
        )

"""Turn text into a humanized CW element timing sequence.

Starts from the ideal PARIS-standard timing (`morse.timing`) and layers on
controlled "sloppy fist" humanization so the resulting training data spans
the failure modes real hand-sent CW exhibits: speed drift mid-message,
per-element duration jitter, and occasional truncated inter-character
spacing (which can make two characters read as a different, longer one --
e.g. a rushed "H E" collapsing toward "5"). The *label* always stays the
original ground-truth text regardless of how aggressively the timing is
humanized; that mismatch between "what the timing looks like" and "what it
actually is" is exactly what a CTC-trained sequence model is meant to learn
to resolve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from morse.morse_code import text_to_morse
from morse.timing import ElementTiming

EventKind = Literal["tone", "silence"]

# Random-walk WPM drift is clamped to this fraction of the base WPM so it
# can never wander into an unreasonable speed.
_MAX_WPM_DRIFT_FRACTION = 0.5


@dataclass(frozen=True)
class Event:
    """A single keyed element: either a tone (dot/dash) or a silence (gap)."""

    kind: EventKind
    duration: float  # seconds, always > 0

    def __post_init__(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"Event duration must be positive, got {self.duration}")


def total_duration(events: list[Event]) -> float:
    """Sum of all event durations, in seconds."""
    return sum(e.duration for e in events)


def _jittered(value: float, std: float, rng: np.random.Generator, min_fraction: float = 0.2) -> float:
    """Apply multiplicative Gaussian jitter to `value`, floored at `min_fraction * value`."""
    if std <= 0:
        return value
    factor = 1.0 + rng.normal(0.0, std)
    factor = max(factor, min_fraction)
    return value * factor


def key_text(
    text: str,
    wpm: float,
    *,
    wpm_drift_std: float = 0.0,
    duration_jitter_std: float = 0.0,
    space_truncation_prob: float = 0.0,
    leading_silence_s: float = 0.0,
    trailing_silence_s: float = 0.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> list[Event]:
    """Render `text` (words separated by single spaces) into a list of `Event`s.

    Args:
        text: message text, A-Z/0-9 and spaces only (spaces become word gaps).
        wpm: base words-per-minute speed.
        wpm_drift_std: relative std-dev of a per-character random-walk applied
            to the effective WPM, simulating an operator speeding up/slowing
            down mid-message. 0 disables drift (constant WPM).
        duration_jitter_std: relative std-dev of per-element Gaussian jitter
            applied to every dot/dash/gap duration. 0 disables jitter.
        space_truncation_prob: probability that an inter-character gap is
            collapsed down to an intra-character gap instead, simulating a
            rushed operator merging two characters. 0 disables truncation.
        leading_silence_s / trailing_silence_s: fixed silence padding appended
            before/after the message (not jittered).
        rng: an existing numpy Generator to draw randomness from; if omitted,
            one is created from `seed` (or from fresh entropy if both are None).

    Returns:
        A list of `Event`s whose durations sum to the total clip duration.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    words = text.upper().split(" ")
    base_timing = ElementTiming.at(wpm)
    current_wpm = wpm

    events: list[Event] = []
    if leading_silence_s > 0:
        events.append(Event("silence", leading_silence_s))

    for word_idx, word in enumerate(words):
        if not word:
            continue
        codes = text_to_morse(word)
        for char_idx, code in enumerate(codes):
            if wpm_drift_std > 0:
                current_wpm += rng.normal(0.0, wpm_drift_std) * wpm
                current_wpm = min(
                    max(current_wpm, wpm * (1 - _MAX_WPM_DRIFT_FRACTION)),
                    wpm * (1 + _MAX_WPM_DRIFT_FRACTION),
                )
            timing = ElementTiming.at(current_wpm) if wpm_drift_std > 0 else base_timing

            for symbol_idx, symbol in enumerate(code):
                dur = timing.dot if symbol == "." else timing.dash
                events.append(Event("tone", _jittered(dur, duration_jitter_std, rng)))
                if symbol_idx < len(code) - 1:
                    events.append(
                        Event("silence", _jittered(timing.intra_char_gap, duration_jitter_std, rng))
                    )

            is_last_char_in_word = char_idx == len(codes) - 1
            if not is_last_char_in_word:
                gap = timing.intra_char_gap if rng.random() < space_truncation_prob else timing.inter_char_gap
                events.append(Event("silence", _jittered(gap, duration_jitter_std, rng)))

        is_last_word = word_idx == len(words) - 1
        if not is_last_word:
            events.append(Event("silence", _jittered(base_timing.word_gap, duration_jitter_std, rng)))

    if trailing_silence_s > 0:
        events.append(Event("silence", trailing_silence_s))

    return events

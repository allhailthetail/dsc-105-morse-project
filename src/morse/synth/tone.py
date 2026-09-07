"""Render a keyed `Event` sequence (from `morse.synth.keyer`) to an audio waveform.

Produces a keyed sine-wave carrier with raised-cosine envelope shaping (to
avoid audible "clicks" at keying transitions) and an optional slow random-walk
frequency drift across the whole message, simulating a radio's oscillator
wandering slightly over the course of a transmission.
"""
from __future__ import annotations

import numpy as np

from morse.synth.keyer import Event

DEFAULT_ENVELOPE_MS = 5.0


def key_to_audio(
    events: list[Event],
    sample_rate: int,
    *,
    freq_hz: float = 600.0,
    freq_drift_hz: float = 0.0,
    envelope_ms: float = DEFAULT_ENVELOPE_MS,
    amplitude: float = 1.0,
    amplitude_jitter_std: float = 0.0,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Render `events` to a mono float32 waveform at `sample_rate`.

    Args:
        freq_hz: base tone carrier frequency.
        freq_drift_hz: std-dev (Hz) of a per-tone-segment random-walk step
            applied to the carrier frequency, so it drifts slowly across the
            whole message rather than resetting per element. 0 disables drift.
        envelope_ms: raised-cosine rise/fall time applied to each tone segment,
            clamped to at most half the segment's duration. 0 gives a hard-keyed
            (instant on/off) tone -- useful for testing click-sensitive code and
            for including some "harder keyed" examples in training data.
        amplitude: peak tone amplitude (silence is always exactly 0).
        amplitude_jitter_std: relative std-dev of per-tone-segment amplitude
            jitter, simulating inconsistent keying pressure/volume.
        rng: existing Generator to draw randomness from; created from `seed`
            (or fresh entropy) if omitted. Unused (deterministic) if both
            freq_drift_hz and amplitude_jitter_std are 0.

    Returns:
        1-D float32 numpy array, `sum(e.duration for e in events) * sample_rate`
        samples long (rounded per-segment).
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    segments: list[np.ndarray] = []
    current_freq = freq_hz
    phase = 0.0  # carried across tone segments so frequency changes don't click

    for event in events:
        n_samples = max(1, round(event.duration * sample_rate))

        if event.kind == "silence":
            segments.append(np.zeros(n_samples, dtype=np.float32))
            continue

        if freq_drift_hz > 0:
            current_freq += rng.normal(0.0, freq_drift_hz)
            current_freq = max(current_freq, 50.0)  # keep it physically sane

        t = np.arange(n_samples) / sample_rate
        instantaneous_phase = phase + 2 * np.pi * current_freq * t
        segment = np.sin(instantaneous_phase).astype(np.float32)
        phase = float(instantaneous_phase[-1] + 2 * np.pi * current_freq / sample_rate)

        seg_amplitude = amplitude
        if amplitude_jitter_std > 0:
            seg_amplitude *= max(1.0 + rng.normal(0.0, amplitude_jitter_std), 0.2)

        envelope = _raised_cosine_envelope(n_samples, sample_rate, envelope_ms)
        segments.append(segment * envelope * seg_amplitude)

    if not segments:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(segments)


def _raised_cosine_envelope(n_samples: int, sample_rate: int, envelope_ms: float) -> np.ndarray:
    """A flat-topped envelope of length `n_samples` with raised-cosine rise/fall."""
    if envelope_ms <= 0 or n_samples < 2:
        return np.ones(n_samples, dtype=np.float32)

    ramp_len = min(round(envelope_ms / 1000.0 * sample_rate), n_samples // 2)
    if ramp_len < 1:
        return np.ones(n_samples, dtype=np.float32)

    envelope = np.ones(n_samples, dtype=np.float32)
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(ramp_len) / ramp_len))
    envelope[:ramp_len] = ramp
    envelope[n_samples - ramp_len:] = ramp[::-1]
    return envelope

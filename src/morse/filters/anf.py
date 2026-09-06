"""Automatic notch filter (ANF): auto-detect a CW tone's center frequency and
bandpass-filter around it. Ported from functions.org's `apply_anf_filter`.

The original R code computed the spectrum peak in kHz and multiplied by 1000
to get Hz; that conversion is specific to `seewave::spec`'s normalized output
and does not apply here -- `numpy.fft.rfftfreq` already returns Hz directly,
so no such conversion is needed (or correct) in this port.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from morse.filters.bandpass import bandpass_filter
from morse.io.wav import read_wav_raw, write_wav

DEFAULT_WINDOW_S = (0.5, 3.0)


def detect_center_frequency(
    samples: np.ndarray,
    sample_rate: int,
    *,
    window_s: tuple[float, float] = DEFAULT_WINDOW_S,
) -> float:
    """Find the dominant spectral peak frequency within `window_s` of `samples`."""
    start = int(window_s[0] * sample_rate)
    end = min(int(window_s[1] * sample_rate), len(samples))
    if start >= end:
        start, end = 0, len(samples)
    segment = samples[start:end]

    spectrum = np.abs(np.fft.rfft(segment))
    freqs = np.fft.rfftfreq(len(segment), d=1.0 / sample_rate)

    peak_index = int(np.argmax(spectrum))
    return float(freqs[peak_index])


def apply_auto_notch_filter(
    file: Path | str,
    output_dir: Path | str,
    *,
    bandwidth_hz: float = 70.0,
    window_s: tuple[float, float] = DEFAULT_WINDOW_S,
) -> Path:
    """Read `file`, auto-detect its CW tone center frequency, bandpass around it, and write out."""
    file = Path(file)
    samples, sample_rate = read_wav_raw(file)

    center_hz = detect_center_frequency(samples, sample_rate, window_s=window_s)
    filtered = bandpass_filter(samples, sample_rate, center_hz=center_hz, bandwidth_hz=bandwidth_hz)

    output_path = Path(output_dir) / f"{file.stem}_anf.wav"
    write_wav(output_path, filtered, sample_rate)
    return output_path

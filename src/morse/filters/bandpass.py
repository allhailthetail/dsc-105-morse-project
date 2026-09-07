"""Butterworth bandpass filtering, ported from functions.org's `apply_bw_filter`.

Uses a zero-phase (`sosfiltfilt`) 2nd-order-section Butterworth filter rather
than a literal transliteration of `seewave::bwfilter` -- this avoids the phase
distortion/edge artifacts a forward-only filter would introduce, at the cost
of not being bit-identical to the original R output (documented tradeoff, see
legacy/archive/main.org for the R behavior this replaces).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

from morse.io.wav import read_wav_raw, write_wav

DEFAULT_ORDER = 4


def bandpass_filter(
    samples: np.ndarray,
    sample_rate: int,
    *,
    center_hz: float = 610.0,
    bandwidth_hz: float = 70.0,
    order: int = DEFAULT_ORDER,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter centered at `center_hz` +/- `bandwidth_hz/2`."""
    low = max(center_hz - bandwidth_hz / 2, 1.0)
    high = min(center_hz + bandwidth_hz / 2, sample_rate / 2 - 1.0)
    sos = butter(order, [low, high], btype="band", fs=sample_rate, output="sos")
    return sosfiltfilt(sos, samples).astype(np.float32)


def apply_bandpass_filter(
    file: Path | str,
    output_dir: Path | str,
    *,
    center_hz: float = 610.0,
    bandwidth_hz: float = 70.0,
    order: int = DEFAULT_ORDER,
) -> Path:
    """Read `file`, bandpass filter it, and write the result to `output_dir`."""
    file = Path(file)
    samples, sample_rate = read_wav_raw(file)
    filtered = bandpass_filter(samples, sample_rate, center_hz=center_hz, bandwidth_hz=bandwidth_hz, order=order)

    output_path = Path(output_dir) / f"{file.stem}_bworth.wav"
    write_wav(output_path, filtered, sample_rate)
    return output_path

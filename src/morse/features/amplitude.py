"""RMS amplitude and zero-crossing-rate, ported from functions.org's
`rms_amplitudes` and the ZCR piece of `wav_metadata`/`seewave::zcr`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from morse.io.wav import read_wav


def rms_amplitude(samples: np.ndarray) -> float:
    """Root-mean-square amplitude of a (cleaned/normalized) waveform."""
    return float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))


def rms_amplitudes(files: list[Path | str]) -> list[float]:
    """Batch RMS amplitude over a list of WAV files (each read+normalized fresh)."""
    return [rms_amplitude(read_wav(f).samples) for f in files]


def zero_crossing_rate(samples: np.ndarray) -> float:
    """Whole-file zero-crossing rate: fraction of adjacent-sample sign changes.

    Matches `seewave::zcr(wav, wl=NULL)` -- a single scalar over the entire
    signal, as used in functions.org's `wav_metadata`.
    """
    if len(samples) < 2:
        return 0.0
    signs = np.sign(samples)
    signs[signs == 0] = 1  # treat exact zero as a continuation of the previous sign
    crossings = np.count_nonzero(np.diff(signs) != 0)
    return crossings / (len(samples) - 1)


def windowed_zero_crossing_rate(samples: np.ndarray, window: int) -> np.ndarray:
    """Zero-crossing rate computed per non-overlapping window of `window` samples.

    Matches the windowed `seewave::zcr(wav, wl=...)` calls in main.org used for
    the ZCR-over-time comparison plots (e.g. `zcr(cw074)[1:150,]`).
    """
    n_windows = len(samples) // window
    return np.array([
        zero_crossing_rate(samples[i * window:(i + 1) * window])
        for i in range(n_windows)
    ])

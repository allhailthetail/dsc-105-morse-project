"""WAV file reading/writing, replacing the old R `tuneR`-based helpers.

Ported from legacy/archive/functions.org's `wav_remove_NAs` and `read_wav`.
The original R code trusted `tuneR`'s reported bit depth, which turned out to
be wrong for this project's real dataset (it reported 8-bit; the files are
actually 32-bit PCM per direct header inspection) -- `read_wav` here instead
always derives the sample dtype from the file's own header via `soundfile`.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf


@dataclass
class WavData:
    """A single-channel, normalized-to-[-1, 1] waveform plus its original file metadata."""

    samples: np.ndarray  # float32, mono, normalized + centered
    sample_rate: int
    subtype: str  # e.g. "PCM_32", as reported by the file header
    bits_per_sample: int
    channels: int


# soundfile subtype -> bits per sample, for the PCM subtypes this project encounters.
_SUBTYPE_BITS = {
    "PCM_S8": 8, "PCM_U8": 8,
    "PCM_16": 16,
    "PCM_24": 24,
    "PCM_32": 32,
    "FLOAT": 32,
    "DOUBLE": 64,
}


def remove_nans(samples: np.ndarray) -> np.ndarray:
    """Replace non-finite (NaN/Inf) samples with 0, matching wav_remove_NAs in functions.org."""
    cleaned = samples.copy()
    cleaned[~np.isfinite(cleaned)] = 0.0
    return cleaned


def read_wav(path: Path | str) -> WavData:
    """Read a WAV file, remove non-finite samples, and normalize+center to [-1, 1].

    Bit depth/subtype is read from the file's own header (not assumed), fixing
    a discrepancy in the original R analysis where `tuneR`'s reported bit depth
    (8-bit) didn't match the files' actual header (32-bit PCM).
    """
    path = Path(path)
    info = sf.info(str(path))
    raw, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)

    if raw.ndim > 1:
        raw = raw.mean(axis=1)  # downmix to mono, matching the dataset's mono assumption

    cleaned = remove_nans(raw)
    centered = cleaned - np.mean(cleaned)
    peak = np.max(np.abs(centered))
    normalized = (centered / peak if peak > 0 else centered).astype(np.float32)

    return WavData(
        samples=normalized,
        sample_rate=sample_rate,
        subtype=info.subtype,
        bits_per_sample=_SUBTYPE_BITS.get(info.subtype, -1),
        channels=info.channels,
    )


def read_wav_raw(path: Path | str) -> tuple[np.ndarray, int]:
    """Read a WAV file's raw (non-normalized) samples and sample rate.

    Used where the original R code deliberately used `readWave` instead of the
    cleaned/normalized `read_wav` (e.g. spectrogram plotting).
    """
    path = Path(path)
    raw, sample_rate = sf.read(str(path), dtype="float64", always_2d=False)
    if raw.ndim > 1:
        raw = raw.mean(axis=1)
    return remove_nans(raw), sample_rate


def write_wav(path: Path | str, samples: np.ndarray, sample_rate: int) -> None:
    """Write mono float samples to a 32-bit float WAV file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), samples, sample_rate, subtype="FLOAT")

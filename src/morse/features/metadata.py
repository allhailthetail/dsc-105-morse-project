"""Per-file WAV metadata, ported from functions.org's `wav_metadata`/`wav_metadata_multi`.

Field names match the original R output for continuity with the archived
main.org analysis, except `bitdepth`, which here correctly reflects the file's
actual header value (see morse.io.wav for why the old R figure was wrong).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from morse.features.amplitude import zero_crossing_rate
from morse.io.wav import read_wav


def wav_metadata(path: Path | str) -> dict:
    """Metadata for a single WAV file: samples, duration, amplitude stats, zcr, format info."""
    wav = read_wav(path)
    samples = wav.samples
    return {
        "filename": str(path),
        "samples": len(samples),
        "duration": round(len(samples) / wav.sample_rate, 3),
        "minAmp": float(samples.min()) if len(samples) else 0.0,
        "medAmp": float(pd.Series(samples).median()) if len(samples) else 0.0,
        "maxAmp": float(samples.max()) if len(samples) else 0.0,
        "zcr": zero_crossing_rate(samples),
        "sampleRate": wav.sample_rate,
        "stereo": wav.channels > 1,
        "pcm": wav.subtype.startswith("PCM"),
        "bitdepth": wav.bits_per_sample,
    }


def wav_metadata_multi(files: list[Path | str]) -> pd.DataFrame:
    """Batch `wav_metadata` over a list of files, as a DataFrame (one row per file)."""
    return pd.DataFrame([wav_metadata(f) for f in files])

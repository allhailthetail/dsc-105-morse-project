#!/usr/bin/env python
"""Reference/QC pass over the archived real CW dataset (data/real/).

Ports the metadata/RMS/ZCR portion of the old main.org EDA. Not part of the
synthetic training-data path -- this only exists to (a) sanity-check the
Python DSP port against the numeric values recorded in main.org's #+RESULTS
blocks, and (b) let data/real/ serve as a held-out real-world eval set later.

Usage:
    python scripts/analyze_real_data.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from morse.features.amplitude import rms_amplitudes
from morse.features.metadata import wav_metadata_multi

DATA_DIR = Path("data/real")


def main() -> None:
    files = sorted(DATA_DIR.glob("cw*.wav"))
    print(f"found {len(files)} real CW recordings in {DATA_DIR}")

    metadata = wav_metadata_multi(files)
    metadata["rms_amplitude"] = rms_amplitudes(files)
    metadata["basename"] = metadata["filename"].apply(lambda p: Path(p).name)

    labels = pd.read_csv(DATA_DIR / "labels.csv")
    merged = metadata.merge(
        labels[["filename", "transcription"]].rename(columns={"filename": "basename"}),
        on="basename",
        how="left",
    )

    print(merged[["filename", "samples", "duration", "bitdepth", "zcr", "rms_amplitude"]].describe(include="all"))
    print(f"\nlabeled examples: {merged['transcription'].notna().sum()} / {len(merged)}")

    print("\nSpot-check against legacy/archive/main.org's recorded #+RESULTS values:")
    print("  cw001.wav expected samples=139046 duration=17.381")
    print("  cw002.wav expected samples=92953 duration=11.619")
    for name in ("cw001.wav", "cw002.wav"):
        row = metadata[metadata["filename"].str.endswith(name)]
        if not row.empty:
            print(f"  {name} actual: samples={row['samples'].iloc[0]} duration={row['duration'].iloc[0]}")


if __name__ == "__main__":
    main()

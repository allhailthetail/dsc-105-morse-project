#!/usr/bin/env python
"""Render a fixed-seed batch of synthetic (wav, label) examples to disk.

This produces a reproducible dataset -- independent of the live tf.data
streaming pipeline (morse.data.pipeline) used for training -- suitable as a
fixed validation/test split. Does not require TensorFlow/Keras.

Usage:
    python scripts/generate_dataset.py --n 200 --seed 0 --out datasets/val
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from morse.io.wav import write_wav
from morse.synth.generator import generate_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100, help="number of examples to render")
    parser.add_argument("--seed", type=int, default=0, help="base seed for reproducibility")
    parser.add_argument("--out", type=Path, default=Path("datasets/generated"), help="output directory")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    labels_path = args.out / "labels.csv"

    with labels_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "transcription", "wpm", "snr_db", "tone_freq_hz"])

        for i in range(args.n):
            rng = np.random.default_rng([args.seed, i])
            example = generate_example(rng=rng)
            filename = f"synth{i:05d}.wav"
            write_wav(args.out / filename, example.audio, example.sample_rate)
            writer.writerow([filename, example.label, round(example.wpm, 2), round(example.snr_db, 2), round(example.tone_freq_hz, 1)])

    print(f"wrote {args.n} examples + labels.csv to {args.out}")


if __name__ == "__main__":
    main()

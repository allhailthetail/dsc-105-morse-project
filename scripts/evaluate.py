#!/usr/bin/env python
"""Evaluate a trained checkpoint's CER/exact-match on synthetic and real CW data.

Reports two numbers side by side, deliberately: mean CER on a fixed synthetic
set (did it learn the synthetic distribution?) and on data/real/'s 58 labeled
real recordings (does it generalize to real audio? -- this is the number
that actually matters). Per the chosen strategy for this phase: train on
synthetic data as-is, run this script, and use the gap between the two
numbers to decide whether the generator's humanization/noise parameters need
revisiting -- don't tune them blind ahead of time.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.weights.h5
    python scripts/evaluate.py --checkpoint checkpoints/best.weights.h5 --synthetic-dir datasets/val
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from morse.eval.metrics import exact_match_rate, mean_character_error_rate
from morse.features.melspec import N_MELS, log_mel_spectrogram
from morse.models.crnn import build_crnn, greedy_decode
from morse.synth.generator import DEFAULT_SAMPLE_RATE, generate_example

# Matches the "test" seed convention in docs/HOWTO-generate-corpus.md.
DEFAULT_SYNTHETIC_SEED = 2
DEFAULT_REAL_DATA_DIR = Path("data/real")


def decode_one(model, features: np.ndarray) -> str:
    batch = features[np.newaxis, ...]
    logits = model(batch, training=False)
    input_length = np.array([features.shape[0]], dtype=np.int32)
    return greedy_decode(logits, input_length)[0]


def evaluate_synthetic(model, n: int, seed: int, sample_rate: int) -> tuple[list[str], list[str]]:
    """Generate `n` fresh synthetic examples in-memory (no disk I/O) and decode each."""
    rng = np.random.default_rng(seed)
    preds, refs = [], []
    for _ in range(n):
        example = generate_example(sample_rate=sample_rate, rng=rng)
        preds.append(decode_one(model, log_mel_spectrogram(example.audio, sample_rate)))
        refs.append(example.label)
    return preds, refs


def evaluate_from_labeled_dir(model, directory: Path, sample_rate: int) -> tuple[list[str], list[str]]:
    """Evaluate against a directory with a labels.csv (filename,transcription): data/real/ or a rendered set."""
    labels = pd.read_csv(directory / "labels.csv")
    preds, refs = [], []
    for _, row in labels.iterrows():
        audio, sr = sf.read(str(directory / row.filename))
        if sr != sample_rate:
            import librosa

            audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=sample_rate)
        preds.append(decode_one(model, log_mel_spectrogram(audio.astype(np.float32), sample_rate)))
        refs.append(row.transcription)
    return preds, refs


def report(name: str, preds: list[str], refs: list[str], show_examples: int = 5) -> None:
    cer = mean_character_error_rate(preds, refs)
    exact = exact_match_rate(preds, refs)
    print(f"\n{name}: n={len(refs)}  mean_CER={cer:.3f}  exact_match={exact:.3f}")
    for p, r in list(zip(preds, refs))[:show_examples]:
        print(f"    pred={p!r:24s} ref={r!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--synthetic-dir", type=Path, default=None,
        help="a dir rendered by scripts/generate_dataset.py; omit to generate in-memory instead",
    )
    parser.add_argument("--synthetic-n", type=int, default=200)
    parser.add_argument("--synthetic-seed", type=int, default=DEFAULT_SYNTHETIC_SEED)
    parser.add_argument("--real-data-dir", type=Path, default=DEFAULT_REAL_DATA_DIR)
    args = parser.parse_args()

    model = build_crnn(n_mels=N_MELS)
    model.load_weights(str(args.checkpoint))

    if args.synthetic_dir is not None:
        preds, refs = evaluate_from_labeled_dir(model, args.synthetic_dir, DEFAULT_SAMPLE_RATE)
    else:
        preds, refs = evaluate_synthetic(model, args.synthetic_n, args.synthetic_seed, DEFAULT_SAMPLE_RATE)
    report("synthetic", preds, refs)

    if (args.real_data_dir / "labels.csv").exists():
        preds, refs = evaluate_from_labeled_dir(model, args.real_data_dir, DEFAULT_SAMPLE_RATE)
        report("real (data/real/)", preds, refs)
    else:
        print(f"\nskipping real-data eval: {args.real_data_dir / 'labels.csv'} not found")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Generate one synthetic example and save its audio + waveform + mel-spectrogram plots.

A quick way to sanity-check the synthetic generator by eye/ear without
touching TensorFlow. Writes to plots/synth_preview/ by default.

Usage:
    python scripts/preview_synth.py --text "CQ CQ DE W1AW" --wpm 20 --snr-db 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from morse.features.melspec import log_mel_spectrogram
from morse.io.wav import write_wav
from morse.synth.generator import generate_example


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", type=str, default=None, help="message text (random if omitted)")
    parser.add_argument("--wpm", type=float, default=None)
    parser.add_argument("--snr-db", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", type=Path, default=Path("plots/synth_preview"))
    args = parser.parse_args()

    example = generate_example(text=args.text, wpm=args.wpm, snr_db=args.snr_db, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    print(f"label:        {example.label}")
    print(f"wpm:          {example.wpm:.1f}")
    print(f"snr_db:       {example.snr_db:.1f}")
    print(f"tone_freq_hz: {example.tone_freq_hz:.1f}")
    print(f"duration_s:   {len(example.audio) / example.sample_rate:.2f}")

    write_wav(args.out / "preview.wav", example.audio, example.sample_rate)

    t = np.arange(len(example.audio)) / example.sample_rate
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, example.audio, color="steelblue", linewidth=0.5)
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title=f"Synthetic CW: {example.label}")
    fig.tight_layout()
    fig.savefig(args.out / "waveform.png", dpi=150)
    plt.close(fig)

    mel = log_mel_spectrogram(example.audio, example.sample_rate)
    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(mel.T, aspect="auto", origin="lower", cmap="magma")
    ax.set(xlabel="Frame", ylabel="Mel bin", title="Log-mel spectrogram")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(args.out / "melspec.png", dpi=150)
    plt.close(fig)

    print(f"wrote preview.wav, waveform.png, melspec.png to {args.out}")


if __name__ == "__main__":
    main()

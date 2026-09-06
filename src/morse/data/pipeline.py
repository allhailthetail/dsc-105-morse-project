"""tf.data.Dataset streaming pipeline over `morse.synth.generator`.

Requires `tensorflow` (see requirements-ml.txt) -- this is the one module in
`morse` that does, since everything else (data generation, features, DSP
utilities) is deliberately kept usable without installing the ML stack.

Produces (log_mel_spectrogram, label_ids) pairs on the fly, with no fixed
on-disk dataset required -- effectively infinite synthetic training data.
For a fixed, reproducible validation/test split, use scripts/generate_dataset.py
instead (renders examples to disk once, with a fixed seed).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import tensorflow as tf

from morse.features.melspec import N_MELS, log_mel_spectrogram
from morse.labels.vocab import encode
from morse.synth.generator import DEFAULT_SAMPLE_RATE, generate_example


def _example_generator(seed: int | None, sample_rate: int) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    while True:
        example = generate_example(sample_rate=sample_rate, rng=rng)
        features = log_mel_spectrogram(example.audio, sample_rate)
        label_ids = encode(example.label)
        yield features, label_ids


def make_dataset(
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
) -> tf.data.Dataset:
    """An infinite `tf.data.Dataset` of (mel_spectrogram, label_ids) pairs, unbatched.

    Shapes are ragged along the time/label-length axes (clips have variable
    duration and labels have variable length) -- batch with `.padded_batch(...)`
    or `.ragged_batch(...)` downstream once a training loop is built.
    """
    return tf.data.Dataset.from_generator(
        lambda: _example_generator(seed, sample_rate),
        output_signature=(
            tf.TensorSpec(shape=(None, N_MELS), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
        ),
    )

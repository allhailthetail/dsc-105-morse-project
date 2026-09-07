"""tf.data.Dataset streaming pipeline over `morse.synth.generator`.

Requires `tensorflow` (see requirements-ml.txt) -- this is the one module in
`morse` that does, since everything else (data generation, features, DSP
utilities) is deliberately kept usable without installing the ML stack.

Produces (features, label_ids, input_length, label_length) tuples on the fly,
with no fixed on-disk dataset required -- effectively infinite synthetic
training data. For a fixed, reproducible validation/test split, use
scripts/generate_dataset.py instead (renders examples to disk once, with a
fixed seed).

The length tensors exist because CTC loss/decoding needs to know how much of
a padded batch element is real content vs. padding -- see
`morse.models.crnn.ctc_loss`/`greedy_decode`, which consume them directly via
`keras.ops.ctc_loss`/`ctc_decode` (the high-level `keras.losses.CTC` class
does NOT accept explicit lengths, so it isn't used here -- see
morse/models/crnn.py's module docstring for why).
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import tensorflow as tf

from morse.features.melspec import N_MELS, log_mel_spectrogram
from morse.labels.vocab import encode
from morse.realtime.config import MAX_MESSAGE_FRAMES
from morse.synth.generator import DEFAULT_SAMPLE_RATE, generate_example

Example = tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]

# Bucket boundaries for `prepare_for_training`'s default bucketed batching --
# see that function's docstring for why this exists at all. Spaced across the
# frame-count distribution real synthetic examples actually land in (empirically
# ~480-2100 frames at default settings: min 480, p50 962, p99 1877, max 2108 over
# a 300-example sample), with the last boundary set just above
# `MAX_MESSAGE_FRAMES` (the ~20s cap `morse.realtime.config` documents as
# comfortable headroom over the generator's own slowest-WPM/longest-message
# default case) so the unbounded overflow bucket is never actually reached at
# default -- or curriculum, which only ever widens WPM *upward* from that same
# floor -- settings.
_DEFAULT_BUCKET_BOUNDARIES: tuple[int, ...] = (
    400, 550, 700, 850, 1000, 1150, 1300, 1500, 1700, 1900, MAX_MESSAGE_FRAMES + 1,
)


def _example_generator(
    seed: int | None,
    sample_rate: int,
    wpm_range: tuple[float, float] | None,
    snr_range_db: tuple[float, float] | None,
    freq_range_hz: tuple[float, float] | None,
    length_range: tuple[int, int] | None,
) -> Iterator[Example]:
    rng = np.random.default_rng(seed)
    kwargs: dict = {}
    if wpm_range is not None:
        kwargs["wpm_range"] = wpm_range
    if snr_range_db is not None:
        kwargs["snr_range_db"] = snr_range_db
    if freq_range_hz is not None:
        kwargs["freq_range_hz"] = freq_range_hz
    if length_range is not None:
        kwargs["length_range"] = length_range

    while True:
        example = generate_example(sample_rate=sample_rate, rng=rng, **kwargs)
        features = log_mel_spectrogram(example.audio, sample_rate)
        label_ids = encode(example.label)
        yield (
            features,
            label_ids,
            np.int32(features.shape[0]),
            np.int32(label_ids.shape[0]),
        )


def make_dataset(
    *,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    seed: int | None = None,
    wpm_range: tuple[float, float] | None = None,
    snr_range_db: tuple[float, float] | None = None,
    freq_range_hz: tuple[float, float] | None = None,
    length_range: tuple[int, int] | None = None,
) -> tf.data.Dataset:
    """An infinite `tf.data.Dataset` of (features, label_ids, input_length, label_length), unbatched.

    `wpm_range`/`snr_range_db`/`freq_range_hz`/`length_range` override
    `generate_example`'s own defaults when given -- use these to implement a
    training curriculum (e.g. widen ranges across training stages) without
    touching scripts/generate_dataset.py's CLI.

    Shapes are ragged along the time/label-length axes (clips have variable
    duration and labels have variable length) -- use `prepare_for_training`
    (or your own `.padded_batch(...)`) downstream.
    """
    return tf.data.Dataset.from_generator(
        lambda: _example_generator(seed, sample_rate, wpm_range, snr_range_db, freq_range_hz, length_range),
        output_signature=(
            tf.TensorSpec(shape=(None, N_MELS), dtype=tf.float32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32),
        ),
    )


def prepare_for_training(
    ds: tf.data.Dataset,
    batch_size: int,
    *,
    drop_remainder: bool = False,
    bucket_boundaries: tuple[int, ...] | None = _DEFAULT_BUCKET_BOUNDARIES,
) -> tf.data.Dataset:
    """Pad and batch a `make_dataset()` stream.

    Padding features with 0.0 (silence-equivalent) and labels with 0 is safe
    even though label id 0 is a real character ("A") -- padded positions are
    never read, because `input_length`/`label_length` are carried alongside
    and consumed explicitly by `morse.models.crnn.ctc_loss`/`greedy_decode`,
    not inferred from padding values.

    By default, batches are grouped by length into `bucket_boundaries`-defined
    buckets (`tf.data.Dataset.bucket_by_sequence_length`) and padded to
    each bucket's own upper boundary, not just to each individual batch's own
    max -- measured empirically on an RTX 3090: with plain `padded_batch`,
    real audio duration varies continuously enough that almost every training
    batch has a distinct time-axis shape, and each *new* shape costs the
    GPU/cuDNN a multi-second one-time re-benchmark that recurs on (nearly)
    every step -- ~14s/step for this small (873K-param) model, almost
    entirely shape-churn overhead rather than actual compute. Bucketing
    collapses the shape space down to `len(bucket_boundaries) + 1` fixed
    shapes that recur throughout training, so that cost is paid a handful of
    times total instead of every step. Pass `bucket_boundaries=None` to fall
    back to plain per-batch padding (e.g. for small one-off eval batches
    where paying for a fixed shape isn't worth it).
    """
    if bucket_boundaries is None:
        return ds.padded_batch(
            batch_size,
            padded_shapes=((None, N_MELS), (None,), (), ()),
            padding_values=(np.float32(0.0), np.int32(0), np.int32(0), np.int32(0)),
            drop_remainder=drop_remainder,
        )

    return ds.bucket_by_sequence_length(
        element_length_func=lambda features, label_ids, input_length, label_length: input_length,
        bucket_boundaries=list(bucket_boundaries),
        bucket_batch_sizes=[batch_size] * (len(bucket_boundaries) + 1),
        # Must be actual TensorShape objects, not plain tuples: passed as
        # tuples, bucket_by_sequence_length's internal shape-substitution
        # (swapping in each bucket's boundary for the padded time axis)
        # mishandles the rank-0 shapes here (input_length/label_length are
        # scalars) and raises "Cannot iterate over a shape with unknown
        # rank" -- confirmed by reading tensorflow/python/data/ops/dataset_
        # ops.py's make_padded_shapes. TensorShape objects sidestep it.
        padded_shapes=(
            tf.TensorShape([None, N_MELS]),
            tf.TensorShape([None]),
            tf.TensorShape([]),
            tf.TensorShape([]),
        ),
        padding_values=(np.float32(0.0), np.int32(0), np.int32(0), np.int32(0)),
        pad_to_bucket_boundary=True,
        drop_remainder=drop_remainder,
    )

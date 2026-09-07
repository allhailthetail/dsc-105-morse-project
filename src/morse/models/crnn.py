"""CRNN (CNN + Bi-LSTM) model producing per-frame CTC logits.

Implements the design recorded in models/README.md. Two things this module
deliberately gets right, worth understanding before changing it:

1. **The conv stack never pools/strides over the time axis** -- only over
   frequency. That means the model's output time dimension always exactly
   equals its input's mel-frame count, so the same `input_length` computed
   by `morse.data.pipeline` can be reused, unchanged, as the CTC
   "output_length"/"logit_length" for both loss and decoding -- no separate
   "how many frames did the conv stack consume" calculation is needed. Don't
   add time-axis pooling/striding without updating that assumption.

2. **`keras.losses.CTC` (the high-level Loss class) is deliberately NOT
   used.** Its functional form (`keras.losses.ctc`) hardcodes
   `mask_index=0` and computes `label_length`/`input_length` itself by
   assuming the *entire* padded time axis is valid content for every
   example -- it has no way to accept the real per-example lengths a padded
   batch actually needs (confirmed by reading its source: see git history/
   session notes for the introspection that established this). This module
   instead calls the lower-level `keras.ops.ctc_loss`/`ctc_decode` directly,
   which do accept explicit lengths and a configurable `mask_index` -- set
   here to `BLANK_INDEX` (the *last* vocab index, per `morse.labels.vocab`),
   not the library's default of 0.

Kept deliberately small (few conv filters, modest LSTM width) since this
needs to run inference on a Raspberry Pi 4 CPU in real time, not just train
well on a GPU -- don't casually grow it without checking Pi4 latency.
"""
from __future__ import annotations

import keras
from keras import layers

from morse.labels.vocab import BLANK_INDEX, VOCAB_SIZE, decode

DEFAULT_CONV_FILTERS = (32, 32)
DEFAULT_LSTM_UNITS = 128
DEFAULT_NUM_LSTM_LAYERS = 2


def build_crnn(
    vocab_size: int = VOCAB_SIZE,
    n_mels: int = 40,
    *,
    conv_filters: tuple[int, ...] = DEFAULT_CONV_FILTERS,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    num_lstm_layers: int = DEFAULT_NUM_LSTM_LAYERS,
    time_steps: int | None = None,
    batch_size: int | None = None,
) -> keras.Model:
    """Build the CRNN: Conv2D (freq-only pooling) -> reshape -> BiLSTM x N -> logits.

    Input: `(batch, time, n_mels)` log-mel spectrogram. Output:
    `(batch, time, vocab_size)` per-frame logits -- no softmax;
    `ctc_loss`/`greedy_decode` below consume raw logits directly, matching
    `keras.ops.ctc_loss`'s expected input.

    `time_steps`/`batch_size` default to `None` (dynamic) for training/eval.
    Pass concrete values to build a **fixed-shape clone** for TFLite export
    (see `scripts/export_tflite.py`): a dynamic time axis through a
    Bidirectional LSTM does not convert with TFLite's builtin ops (confirmed
    empirically -- it requires the Flex-ops delegate, which defeats the
    point of the lightweight `tflite-runtime` package on a Pi4), while a
    concrete shape converts cleanly with builtins only. Weights are
    shape-independent (Conv2D/LSTM parameter shapes don't depend on the time
    axis), so a fixed-shape clone can load the exact weights trained with
    the dynamic-shape version via `get_weights()`/`set_weights()`.
    """
    inputs = keras.Input(shape=(time_steps, n_mels), batch_size=batch_size, name="mel_spectrogram")
    x = layers.Reshape((-1, n_mels, 1))(inputs)  # add a channel dim for Conv2D

    for filters in conv_filters:
        x = layers.Conv2D(filters, kernel_size=3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPooling2D(pool_size=(1, 2))(x)  # (time, freq) -- freq only

    # Collapse (freq', channels) into one feature axis per time step. Both are
    # static/known at graph-build time (only the time axis is dynamic), so
    # this Reshape's target shape is concrete.
    freq_dim, channels = x.shape[-2], x.shape[-1]
    x = layers.Reshape((-1, freq_dim * channels))(x)

    for _ in range(num_lstm_layers):
        x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)

    # Dense already applies per-timestep over the last axis in Keras/TF, so no
    # TimeDistributed wrapper is needed.
    logits = layers.Dense(vocab_size, name="logits")(x)

    return keras.Model(inputs=inputs, outputs=logits, name="morse_crnn")


def ctc_loss(labels, logits, label_length, input_length):
    """Per-example CTC loss. Shapes: labels (B,L), logits (B,T,V), lengths (B,).

    Thin wrapper over `keras.ops.ctc_loss` fixing `mask_index=BLANK_INDEX` to
    match this project's vocabulary convention (blank is the *last* index,
    not index 0 -- see module docstring).
    """
    return keras.ops.ctc_loss(labels, logits, label_length, input_length, mask_index=BLANK_INDEX)


def greedy_decode(logits, input_length) -> list[str]:
    """Greedy CTC decode: per-frame logits -> a list of decoded text strings.

    Shapes: logits (B,T,V), input_length (B,). Wraps `keras.ops.ctc_decode`
    (strategy="greedy") with `mask_index=BLANK_INDEX`, then
    `morse.labels.vocab.decode` to turn the resulting id sequences into text.
    """
    decoded, _ = keras.ops.ctc_decode(
        logits, sequence_lengths=input_length, strategy="greedy", mask_index=BLANK_INDEX
    )
    decoded = keras.ops.convert_to_numpy(decoded[0])  # (batch, max_length), -1 = blank/pad

    texts = []
    for row in decoded:
        ids = [int(i) for i in row if i >= 0]
        texts.append(decode(ids))
    return texts

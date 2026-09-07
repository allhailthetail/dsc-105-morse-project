# Model

Implemented in [`crnn.py`](crnn.py): `build_crnn()`, plus `ctc_loss()`/
`greedy_decode()` helpers. Training itself lives in
[`scripts/train.py`](../../../scripts/train.py), not here.

## Target architecture: CRNN + CTC

A convolutional-recurrent network trained with CTC (Connectionist Temporal
Classification) loss, not a plain CNN classifier — chosen because naive
threshold/timing-based Morse decoders fail on real hand-sent CW:

- **Rigid timing assumptions break on real operators.** Hand-sent CW doesn't
  hold a strict dit:dah:space ratio — WPM drifts mid-message, inter-character
  spaces get truncated ("sloppy fist," e.g. a rushed `H E` collapsing toward
  what looks like `5`), and simple decoders have no error correction to
  recover from a single misread gap.
- **Convolutional layers** learn the fuzzy visual "shape" of a dit/dah in the
  mel-spectrogram rather than a hard amplitude threshold, so they stay robust
  when a signal fades mid-element (QSB) instead of failing outright.
- **A bidirectional LSTM** supplies sequence context in both directions,
  letting the model use surrounding characters to resolve an ambiguous or
  run-together segment — similar to how a human listener uses context.
- **CTC loss** removes the need for frame-level alignment between audio and
  text: it maps a variable-length input sequence (audio frames) to a
  variable-length output (text) without requiring the two to line up
  one-to-one, which is exactly the property needed when sender speed and
  spacing aren't fixed.

## Implemented shape

1. Input: `(batch, time, n_mels)` log-mel spectrogram frames from
   [`morse.features.melspec`](../features/melspec.py) — the same function
   must be used at inference time to avoid a train/deploy preprocessing
   mismatch.
2. A stack of Conv2D blocks (`build_crnn`'s `conv_filters`) for local
   time-frequency feature extraction, pooling along the frequency axis
   **only** — the time axis is never pooled/strided, so the model's output
   time dimension always exactly equals its input time dimension. This is
   relied on directly: the same `input_length` computed by
   `morse.data.pipeline` is reused unchanged as the CTC output length.
3. Reshape the conv output `(time, freq', channels)` into `(time, features)`.
4. `num_lstm_layers` bidirectional LSTM layers (`lstm_units` wide) for
   temporal context.
5. A `Dense(vocab_size)` producing per-frame **logits** (no softmax — CTC
   loss/decode consume raw logits) over `VOCAB_SIZE` (see
   [`morse.labels.vocab`](../labels/vocab.py): 36 characters + 1 CTC blank,
   blank at the *last* index).
6. Training/inference use `keras.ops.ctc_loss`/`keras.ops.ctc_decode`
   directly (via `ctc_loss()`/`greedy_decode()` in `crnn.py`), **not** the
   high-level `keras.losses.CTC` class — that class hardcodes
   `mask_index=0` and infers lengths by assuming the entire padded time axis
   is valid, which breaks on variable-length padded batches. The lower-level
   ops accept explicit `input_length`/`label_length` (from
   `morse.data.pipeline`) and a configurable `mask_index=BLANK_INDEX`.

Deliberately kept small (few conv filters, modest LSTM width) — this needs to
run inference on a Raspberry Pi 4 CPU in real time, not just train well on a
GPU. Don't grow it without checking Pi4 latency.

## Deployment target

Raspberry Pi 4 + a cheap microphone, capturing environmental noise plus CW
audio. The embedded pipeline pre-processes raw audio into a mel-spectrogram
(via `morse.features.melspec`) before it reaches the model — this is why that
module is the single source of truth for preprocessing parameters, not
something bundled into training-only code.

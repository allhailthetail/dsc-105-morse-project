# Model (placeholder — not implemented yet)

This package intentionally contains no model code yet. Building and training
the model is near-future work, out of scope for the current data/preprocessing
migration. This file records the target design so the decision isn't lost.

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

## Planned shape (not yet implemented)

1. Input: `(time, n_mels)` log-mel spectrogram frames from
   [`morse.features.melspec`](../features/melspec.py) — the same function
   must be used at inference time to avoid a train/deploy preprocessing
   mismatch.
2. A stack of Conv2D (or Conv1D-over-mel-bins) blocks for local
   time-frequency feature extraction, with pooling along the frequency axis
   only (preserving time resolution for the recurrent stage).
3. Reshape/flatten the conv output into a time-major sequence.
4. One or more bidirectional LSTM layers for temporal context.
5. A dense layer + softmax over `VOCAB_SIZE` (see
   [`morse.labels.vocab`](../labels/vocab.py): 36 characters + 1 CTC blank).
6. CTC loss (`keras.losses` / `tf.nn.ctc_loss`) for training; CTC greedy or
   beam-search decoding (`morse.labels.vocab.ctc_collapse` handles the
   greedy case) for inference.

## Deployment target

Raspberry Pi 4 + a cheap microphone, capturing environmental noise plus CW
audio. The embedded pipeline pre-processes raw audio into a mel-spectrogram
(via `morse.features.melspec`) before it reaches the model — this is why that
module is the single source of truth for preprocessing parameters, not
something bundled into training-only code.

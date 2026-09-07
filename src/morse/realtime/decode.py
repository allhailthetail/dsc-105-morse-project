"""Run a finalized audio segment through the exported TFLite model and decode it.

Deliberately avoids depending on Keras/`keras.ops.ctc_decode` here (unlike
`morse.models.crnn.greedy_decode`, used for training-time evaluation) --
`morse.labels.vocab.ctc_collapse` is pure numpy/python, so this module only
needs a TFLite interpreter, not a full ML framework. This matters for the
actual Pi4 deployment: it should be able to run with just the lightweight
`tflite-runtime` package rather than full `tensorflow`.
"""
from __future__ import annotations

import numpy as np

from morse.features.melspec import N_MELS, log_mel_spectrogram
from morse.labels.vocab import ctc_collapse
from morse.realtime.config import MAX_MESSAGE_FRAMES

try:  # pragma: no cover -- exercised on whichever runtime is actually installed
    from tflite_runtime.interpreter import Interpreter  # type: ignore[import-not-found]
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def load_interpreter(model_path: str):
    """Load a TFLite model and allocate its tensors, ready for repeated inference."""
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def _pad_or_trim(features: np.ndarray, target_frames: int) -> np.ndarray:
    """Pad with 0.0 (silence) or trim `features` (time, n_mels) to exactly `target_frames`."""
    n_frames = features.shape[0]
    if n_frames == target_frames:
        return features
    if n_frames > target_frames:
        return features[:target_frames]
    padded = np.zeros((target_frames, features.shape[1]), dtype=features.dtype)
    padded[:n_frames] = features
    return padded


def decode_segment(
    raw_audio: np.ndarray,
    sample_rate: int,
    interpreter,
    *,
    max_frames: int = MAX_MESSAGE_FRAMES,
) -> str:
    """Preprocess + run inference + greedy-decode one finalized audio segment.

    `interpreter` must have been built from a model exported with a fixed
    `time_steps == max_frames` (see scripts/export_tflite.py) -- the segment's
    mel-spectrogram is padded/trimmed to match exactly, since the model's
    input shape is fixed, not dynamic.
    """
    features = log_mel_spectrogram(raw_audio, sample_rate)
    n_real_frames = features.shape[0]
    features = _pad_or_trim(features, max_frames)

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    interpreter.set_tensor(input_details["index"], features[np.newaxis, ...].astype(np.float32))
    interpreter.invoke()
    logits = interpreter.get_tensor(output_details["index"])[0]  # (max_frames, vocab_size)

    # Only decode over the real (non-padded) frames -- padding is silence,
    # which the model should mostly predict as blank anyway, but there's no
    # reason to feed it to the decoder.
    real_logits = logits[: min(n_real_frames, max_frames)]
    ids = np.argmax(real_logits, axis=-1)
    return ctc_collapse(ids)


__all__ = ["N_MELS", "decode_segment", "load_interpreter"]

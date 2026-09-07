"""Log-mel spectrogram feature extraction.

This is the single source of truth for mel-spectrogram parameters, meant to
be shared unchanged between the training data pipeline (this module) and any
future on-device (Raspberry Pi) inference preprocessing -- a mismatch between
training and deployment preprocessing would silently break the model, so any
future embedded reimplementation MUST match these exact parameter values (or
import this module directly, if the deployment target can run Python).
"""
from __future__ import annotations

import numpy as np

# Defaults chosen for 16kHz mono CW-in-noise audio (see morse.synth.generator's
# DEFAULT_SAMPLE_RATE). CW tone content lives around 400-1000Hz, so fmax is set
# generously above that to also capture noise/harmonic context.
SAMPLE_RATE = 16_000
N_FFT = 512
HOP_LENGTH = 160  # 10ms at 16kHz
WIN_LENGTH = 400  # 25ms at 16kHz
N_MELS = 40
FMIN = 50.0
FMAX = 4000.0


def log_mel_spectrogram(
    samples: np.ndarray,
    sample_rate: int = SAMPLE_RATE,
    *,
    n_fft: int = N_FFT,
    hop_length: int = HOP_LENGTH,
    win_length: int = WIN_LENGTH,
    n_mels: int = N_MELS,
    fmin: float = FMIN,
    fmax: float = FMAX,
) -> np.ndarray:
    """Compute a log-scaled mel spectrogram: shape (n_frames, n_mels).

    Time is the first axis (not librosa's default (n_mels, n_frames)) since
    that's the layout a CRNN's recurrent layers expect downstream.
    """
    import librosa  # local import: keeps this module's import cost low when unused

    if sample_rate != SAMPLE_RATE:
        raise ValueError(
            f"melspec params (n_fft/hop_length/win_length) are tuned for {SAMPLE_RATE}Hz, "
            f"got {sample_rate}Hz -- resample first or pass matching sample_rate-scaled params"
        )

    mel = librosa.feature.melspectrogram(
        y=samples.astype(np.float32),
        sr=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax,
        power=2.0,
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.T.astype(np.float32)  # (n_frames, n_mels)


def expected_num_frames(num_samples: int, hop_length: int = HOP_LENGTH) -> int:
    """Number of mel-spectrogram frames librosa produces for a clip of `num_samples`."""
    return 1 + num_samples // hop_length

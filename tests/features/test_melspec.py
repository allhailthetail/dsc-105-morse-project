import numpy as np
import pytest

from morse.features.melspec import HOP_LENGTH, N_MELS, SAMPLE_RATE, expected_num_frames, log_mel_spectrogram


def test_output_shape_matches_expected_frames():
    duration_s = 1.0
    n_samples = int(duration_s * SAMPLE_RATE)
    samples = 0.5 * np.sin(2 * np.pi * 600 * np.arange(n_samples) / SAMPLE_RATE).astype(np.float32)

    mel = log_mel_spectrogram(samples, SAMPLE_RATE)
    assert mel.shape[1] == N_MELS
    assert mel.shape[0] == expected_num_frames(n_samples, HOP_LENGTH)


def test_rejects_mismatched_sample_rate():
    samples = np.zeros(8000, dtype=np.float32)
    with pytest.raises(ValueError):
        log_mel_spectrogram(samples, sample_rate=8000)


def test_deterministic_for_same_input():
    samples = np.random.default_rng(0).normal(0, 1, SAMPLE_RATE).astype(np.float32)
    a = log_mel_spectrogram(samples, SAMPLE_RATE)
    b = log_mel_spectrogram(samples, SAMPLE_RATE)
    assert np.array_equal(a, b)

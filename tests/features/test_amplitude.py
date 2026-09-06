import numpy as np
import pytest

from morse.features.amplitude import rms_amplitude, windowed_zero_crossing_rate, zero_crossing_rate
from morse.io.wav import read_wav


def test_rms_amplitude_of_known_signal():
    # A unit-amplitude sine wave has RMS = 1/sqrt(2).
    n = 16000
    samples = np.sin(2 * np.pi * 440 * np.arange(n) / 16000)
    assert rms_amplitude(samples) == pytest.approx(1 / np.sqrt(2), rel=1e-3)


def test_zero_crossing_rate_of_alternating_signal():
    samples = np.array([1, -1, 1, -1, 1, -1], dtype=np.float32)
    assert zero_crossing_rate(samples) == pytest.approx(1.0)


def test_zero_crossing_rate_of_constant_signal_is_zero():
    samples = np.ones(100, dtype=np.float32)
    assert zero_crossing_rate(samples) == 0.0


def test_windowed_zcr_matches_whole_file_for_single_window():
    samples = np.sin(2 * np.pi * 5 * np.arange(1000) / 1000).astype(np.float32)
    windowed = windowed_zero_crossing_rate(samples, window=len(samples))
    assert len(windowed) == 1
    assert windowed[0] == pytest.approx(zero_crossing_rate(samples))


def test_real_data_cw074_cleaner_than_cw092(real_data_dir):
    # main.org identifies cw074 as the highest-RMS ("clean") file and cw092 as
    # the lowest-RMS ("noisy") file among the real dataset.
    clean_path = real_data_dir / "cw074.wav"
    noisy_path = real_data_dir / "cw092.wav"
    if not (clean_path.exists() and noisy_path.exists()):
        pytest.skip("data/real/ not present")

    clean_rms = rms_amplitude(read_wav(clean_path).samples)
    noisy_rms = rms_amplitude(read_wav(noisy_path).samples)
    assert clean_rms > noisy_rms

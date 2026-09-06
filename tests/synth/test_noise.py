import numpy as np
import pytest

from morse.synth.noise import (
    load_noise_bank,
    measured_snr_db,
    mix_at_snr,
    pink_noise,
    qsb_envelope,
    sample_noise_bed,
    white_noise,
)


def test_mix_at_snr_hits_target(rng):
    n = 16000
    signal = 0.5 * np.sin(2 * np.pi * 600 * np.arange(n) / 16000).astype(np.float32)
    noise = white_noise(n, rng)

    for target_db in (-5.0, 0.0, 5.0, 10.0):
        mixed = mix_at_snr(signal, noise, target_db)
        residual_noise = mixed - signal
        achieved_db = measured_snr_db(signal, residual_noise)
        assert achieved_db == pytest.approx(target_db, abs=0.5)


def test_mix_requires_equal_length():
    with pytest.raises(ValueError):
        mix_at_snr(np.zeros(10), np.zeros(5), 0.0)


def test_white_noise_is_seeded_and_reproducible():
    a = white_noise(100, np.random.default_rng(1))
    b = white_noise(100, np.random.default_rng(1))
    assert np.array_equal(a, b)


def test_pink_noise_has_more_low_frequency_energy_than_white(rng):
    n = 16000 * 2
    white = white_noise(n, np.random.default_rng(0))
    pink = pink_noise(n, np.random.default_rng(0))

    def low_freq_fraction(x):
        spectrum = np.abs(np.fft.rfft(x)) ** 2
        freqs = np.fft.rfftfreq(len(x), d=1 / 16000)
        low = spectrum[freqs < 500].sum()
        return low / spectrum.sum()

    assert low_freq_fraction(pink) > low_freq_fraction(white)


def test_qsb_envelope_stays_in_expected_range(rng):
    env = qsb_envelope(16000, 16000, rng, depth=0.6)
    assert env.min() >= 1.0 - 0.6 - 1e-6
    assert env.max() <= 1.0 + 1e-6


def test_qsb_envelope_depth_zero_is_constant(rng):
    env = qsb_envelope(1000, 16000, rng, depth=0.0)
    assert np.allclose(env, 1.0)


def test_load_noise_bank_finds_categories(noise_data_dir):
    if not noise_data_dir.is_dir():
        pytest.skip("data/noise/ not present")
    bank = load_noise_bank(noise_data_dir, sample_rate=16000)
    assert any(bank.values()), "expected at least one noise category with files"


def test_sample_noise_bed_returns_requested_length(noise_data_dir, rng):
    if not noise_data_dir.is_dir():
        pytest.skip("data/noise/ not present")
    bank = load_noise_bank(noise_data_dir, sample_rate=16000)
    if not any(bank.values()):
        pytest.skip("no noise files found")
    n = 8000
    bed = sample_noise_bed(bank, n, 16000, rng)
    assert len(bed) == n

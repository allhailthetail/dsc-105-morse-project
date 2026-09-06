import numpy as np
import pytest

from morse.synth.keyer import Event
from morse.synth.tone import key_to_audio


def test_output_length_matches_total_duration():
    sample_rate = 16000
    events = [Event("tone", 0.1), Event("silence", 0.05), Event("tone", 0.2)]
    audio = key_to_audio(events, sample_rate)
    expected_samples = round(0.1 * sample_rate) + round(0.05 * sample_rate) + round(0.2 * sample_rate)
    assert len(audio) == expected_samples


def test_silence_segments_are_exactly_zero():
    sample_rate = 8000
    events = [Event("silence", 0.1)]
    audio = key_to_audio(events, sample_rate)
    assert np.all(audio == 0.0)


def test_tone_segment_has_expected_frequency():
    sample_rate = 16000
    freq = 600.0
    events = [Event("tone", 0.5)]
    audio = key_to_audio(events, sample_rate, freq_hz=freq, envelope_ms=0)

    spectrum = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    peak_freq = freqs[np.argmax(spectrum)]
    assert peak_freq == pytest.approx(freq, abs=5.0)


def test_envelope_reduces_click_energy():
    # A global max-derivative comparison doesn't isolate the click: a full-
    # amplitude sine's own inherent per-sample slope can exceed any boundary
    # discontinuity, masking the effect of envelope smoothing. Instead, look
    # specifically at the derivative right at the tone -> silence transition.
    sample_rate = 16000
    tone_duration = 0.2
    events = [Event("tone", tone_duration), Event("silence", 0.05)]
    boundary_index = round(tone_duration * sample_rate)  # last tone sample -> first silence sample

    hard_keyed = key_to_audio(events, sample_rate, envelope_ms=0)
    smoothed = key_to_audio(events, sample_rate, envelope_ms=5.0)

    hard_jump = abs(hard_keyed[boundary_index] - hard_keyed[boundary_index - 1])
    smooth_jump = abs(smoothed[boundary_index] - smoothed[boundary_index - 1])
    assert smooth_jump < hard_jump


def test_amplitude_scales_output():
    sample_rate = 8000
    events = [Event("tone", 0.1)]
    quiet = key_to_audio(events, sample_rate, amplitude=0.1, envelope_ms=0)
    loud = key_to_audio(events, sample_rate, amplitude=1.0, envelope_ms=0)
    assert np.max(np.abs(loud)) > np.max(np.abs(quiet))


def test_freq_drift_is_seeded_and_reproducible():
    sample_rate = 16000
    events = [Event("tone", 0.5), Event("silence", 0.1), Event("tone", 0.5)]
    a = key_to_audio(events, sample_rate, freq_drift_hz=5.0, seed=7)
    b = key_to_audio(events, sample_rate, freq_drift_hz=5.0, seed=7)
    assert np.array_equal(a, b)

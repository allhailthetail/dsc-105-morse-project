import numpy as np
import pytest

from morse.synth.keyer import key_text, total_duration
from morse.timing import ElementTiming


def test_clean_timing_matches_paris_standard():
    wpm = 20
    events = key_text("SOS", wpm)
    timing = ElementTiming.at(wpm)

    # S = ... , O = ---, S = ...
    # 3 dots + 2 intra-gaps (S), inter-gap, 3 dashes + 2 intra-gaps (O), inter-gap, 3 dots + 2 intra-gaps (S)
    expected = (
        3 * timing.dot + 2 * timing.intra_char_gap
        + timing.inter_char_gap
        + 3 * timing.dash + 2 * timing.intra_char_gap
        + timing.inter_char_gap
        + 3 * timing.dot + 2 * timing.intra_char_gap
    )
    assert total_duration(events) == pytest.approx(expected)


def test_word_gap_inserted_between_words():
    wpm = 20
    events_one_word = key_text("SOS", wpm)
    events_two_words = key_text("SOS SOS", wpm)
    timing = ElementTiming.at(wpm)

    expected = 2 * total_duration(events_one_word) + timing.word_gap
    assert total_duration(events_two_words) == pytest.approx(expected)


def test_leading_and_trailing_silence():
    events = key_text("E", 20, leading_silence_s=0.3, trailing_silence_s=0.2)
    assert events[0].kind == "silence"
    assert events[0].duration == pytest.approx(0.3)
    assert events[-1].kind == "silence"
    assert events[-1].duration == pytest.approx(0.2)


def test_jitter_is_seeded_and_reproducible():
    rng1 = np.random.default_rng(123)
    rng2 = np.random.default_rng(123)
    e1 = key_text("SOS", 20, duration_jitter_std=0.2, rng=rng1)
    e2 = key_text("SOS", 20, duration_jitter_std=0.2, rng=rng2)
    assert [e.duration for e in e1] == [e.duration for e in e2]


def test_jitter_bounded_by_min_fraction():
    rng = np.random.default_rng(0)
    # Large jitter std should still never produce non-positive (or absurdly small) durations.
    events = key_text("SOS SOS SOS", 20, duration_jitter_std=5.0, rng=rng)
    timing = ElementTiming.at(20)
    for e in events:
        assert e.duration > 0
        assert e.duration >= 0.2 * min(timing.dot, timing.intra_char_gap) - 1e-9


def test_space_truncation_shortens_all_inter_char_gaps():
    wpm = 20
    timing = ElementTiming.at(wpm)
    rng = np.random.default_rng(1)
    # A single word, so every silence event is either an intra- or inter-character
    # gap (no word gaps). With truncation_prob=1.0, every inter-character gap
    # collapses to the (shorter) intra-character gap duration.
    events = key_text("ABCDEFGHIJKLMNOP", wpm, space_truncation_prob=1.0, rng=rng)
    silence_durations = [e.duration for e in events if e.kind == "silence"]
    assert all(d == pytest.approx(timing.intra_char_gap) for d in silence_durations)

    # Without truncation, some silences should be the (longer) inter-character gap.
    events_no_truncation = key_text("ABCDEFGHIJKLMNOP", wpm, space_truncation_prob=0.0)
    silence_durations_no_truncation = [e.duration for e in events_no_truncation if e.kind == "silence"]
    assert any(d == pytest.approx(timing.inter_char_gap) for d in silence_durations_no_truncation)


def test_wpm_drift_changes_element_durations_over_message():
    rng = np.random.default_rng(2)
    events = key_text("A" * 20, 20, wpm_drift_std=0.1, rng=rng)
    tone_durations = [e.duration for e in events if e.kind == "tone"]
    # Drift should introduce some variation across the message (not all identical).
    assert len(set(round(d, 6) for d in tone_durations)) > 1

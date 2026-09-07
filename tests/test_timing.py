import pytest

from morse.timing import ElementTiming, approx_wpm, tau_seconds


@pytest.mark.parametrize("wpm", [5, 13, 20, 35, 60])
def test_paris_word_takes_60_over_wpm_seconds(wpm):
    # PARIS = 50 tau by definition, and the whole point of the standard is
    # that sending "PARIS " (with trailing word gap) at `wpm` takes 60/wpm sec.
    tau = tau_seconds(wpm)
    assert tau * 50 == pytest.approx(60.0 / wpm)


def test_element_timing_ratios():
    t = ElementTiming.at(20)
    assert t.dash == pytest.approx(3 * t.dot)
    assert t.intra_char_gap == pytest.approx(t.dot)
    assert t.inter_char_gap == pytest.approx(3 * t.dot)
    assert t.word_gap == pytest.approx(7 * t.dot)


def test_faster_wpm_means_shorter_elements():
    slow = ElementTiming.at(10)
    fast = ElementTiming.at(40)
    assert fast.dot < slow.dot
    assert fast.dash < slow.dash


def test_tau_seconds_rejects_non_positive_wpm():
    with pytest.raises(ValueError):
        tau_seconds(0)


def test_approx_wpm_matches_main_org_heuristic():
    # main.org: approx_wpm = round((12 * numChars) / duration, 2), numChars fixed at 20
    assert approx_wpm(20, 17.381) == pytest.approx(round(12 * 20 / 17.381, 2))
    assert approx_wpm(20, 11.619) == pytest.approx(round(12 * 20 / 11.619, 2))

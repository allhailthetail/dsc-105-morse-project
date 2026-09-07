import pytest

from morse.eval.metrics import (
    character_error_rate,
    exact_match,
    exact_match_rate,
    mean_character_error_rate,
)


def test_identical_strings_have_zero_cer():
    assert character_error_rate("SOS", "SOS") == 0.0


def test_single_substitution():
    assert character_error_rate("SOT", "SOS") == pytest.approx(1 / 3)


def test_single_insertion():
    assert character_error_rate("SOSX", "SOS") == pytest.approx(1 / 3)


def test_single_deletion():
    assert character_error_rate("SO", "SOS") == pytest.approx(1 / 3)


def test_completely_different_strings():
    assert character_error_rate("ABC", "XYZ") == pytest.approx(1.0)


def test_empty_ref_and_empty_pred():
    assert character_error_rate("", "") == 0.0


def test_empty_ref_nonempty_pred():
    assert character_error_rate("ABC", "") == 3.0


def test_exact_match():
    assert exact_match("SOS", "SOS") is True
    assert exact_match("SOS", "SOT") is False


def test_mean_cer_over_list():
    preds = ["SOS", "ABC"]
    refs = ["SOS", "ABX"]
    assert mean_character_error_rate(preds, refs) == pytest.approx((0.0 + 1 / 3) / 2)


def test_mean_cer_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        mean_character_error_rate(["A"], ["A", "B"])


def test_exact_match_rate():
    preds = ["SOS", "ABC", "XYZ"]
    refs = ["SOS", "ABC", "QQQ"]
    assert exact_match_rate(preds, refs) == pytest.approx(2 / 3)


def test_empty_lists():
    assert mean_character_error_rate([], []) == 0.0
    assert exact_match_rate([], []) == 0.0

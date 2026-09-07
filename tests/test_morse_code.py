import pytest

from morse.morse_code import ALPHABET, MORSE_TABLE, morse_to_text, text_to_morse


def test_alphabet_covers_a_to_z_and_0_to_9():
    assert set(ALPHABET) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
    assert len(ALPHABET) == 36


@pytest.mark.parametrize("char", list(MORSE_TABLE.keys()))
def test_round_trip_single_char(char):
    codes = text_to_morse(char)
    assert morse_to_text(codes) == char


def test_round_trip_word():
    text = "SOS73"
    assert morse_to_text(text_to_morse(text)) == text


def test_lowercase_is_normalized():
    assert text_to_morse("sos") == text_to_morse("SOS")


def test_unsupported_character_raises():
    with pytest.raises(ValueError):
        text_to_morse("HELLO WORLD!")


def test_known_codes():
    assert MORSE_TABLE["S"] == "..."
    assert MORSE_TABLE["O"] == "---"
    assert MORSE_TABLE["A"] == ".-"

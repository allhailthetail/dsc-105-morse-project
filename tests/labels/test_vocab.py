import pytest

from morse.labels.vocab import BLANK_INDEX, VOCAB_SIZE, ctc_collapse, decode, encode
from morse.morse_code import ALPHABET


def test_vocab_size_is_alphabet_plus_blank():
    assert VOCAB_SIZE == len(ALPHABET) + 1
    assert BLANK_INDEX == len(ALPHABET)


def test_encode_decode_round_trip_full_alphabet():
    assert decode(encode(ALPHABET)) == ALPHABET


@pytest.mark.parametrize("text", ["SOS", "CQ", "73", "HELLOWORLD"])
def test_encode_decode_round_trip(text):
    assert decode(encode(text)) == text


def test_encode_rejects_unsupported_characters():
    with pytest.raises(ValueError):
        encode("HELLO WORLD")  # space not in vocab


def test_encoded_ids_never_include_blank():
    ids = encode(ALPHABET)
    assert BLANK_INDEX not in ids


def test_ctc_collapse_merges_repeats_and_drops_blanks():
    # Raw per-frame output: S-S-blank-blank-O-O-blank-S-S (blank = BLANK_INDEX)
    s = encode("S")[0]
    o = encode("O")[0]
    b = BLANK_INDEX
    raw = [s, s, b, b, o, o, b, s, s]
    assert ctc_collapse(raw) == "SOS"


def test_ctc_collapse_of_empty_sequence():
    assert ctc_collapse([]) == ""

import pytest

from morse.features.metadata import wav_metadata


# Reference values from legacy/archive/main.org's `gather_basic_metadata` #+RESULTS block.
# Note: bitdepth is EXPECTED to differ from the R output (8) -- the R analysis mis-reported
# bit depth via tuneR; direct header inspection confirms these files are actually 32-bit PCM.
# See morse.io.wav's module docstring for details.
EXPECTED = {
    "cw001.wav": {"samples": 139046, "duration": 17.381},
    "cw002.wav": {"samples": 92953, "duration": 11.619},
}


@pytest.mark.parametrize("filename", list(EXPECTED.keys()))
def test_metadata_matches_archived_r_results(real_data_dir, filename):
    path = real_data_dir / filename
    if not path.exists():
        pytest.skip("data/real/ not present")

    meta = wav_metadata(path)
    expected = EXPECTED[filename]
    assert meta["samples"] == expected["samples"]
    assert meta["duration"] == pytest.approx(expected["duration"])
    assert meta["bitdepth"] == 32
    assert meta["stereo"] is False
    assert meta["sampleRate"] == 8000

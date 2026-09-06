from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def real_data_dir() -> Path:
    return REPO_ROOT / "data" / "real"


@pytest.fixture
def noise_data_dir() -> Path:
    return REPO_ROOT / "data" / "noise"


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def tmp_wav_file(tmp_path):
    """A small synthetic sine-wave WAV file for fast I/O unit tests."""
    import soundfile as sf

    sample_rate = 8000
    t = np.arange(sample_rate) / sample_rate  # 1 second
    samples = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    path = tmp_path / "test_tone.wav"
    sf.write(str(path), samples, sample_rate, subtype="PCM_32")
    return path, samples, sample_rate

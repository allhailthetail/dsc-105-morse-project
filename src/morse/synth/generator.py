"""End-to-end synthetic example generation: random/given text -> noisy CW audio + label.

This is the main entry point the rest of the project (dataset rendering,
the tf.data pipeline, preview scripts) should use rather than composing
`keyer`/`tone`/`noise` by hand.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from morse.morse_code import ALPHABET
from morse.synth.keyer import key_text, total_duration
from morse.synth.noise import load_noise_bank, mix_at_snr, pink_noise, qsb_envelope, sample_noise_bed, white_noise
from morse.synth.tone import key_to_audio

DEFAULT_SAMPLE_RATE = 16_000
DEFAULT_NOISE_DIR = Path(__file__).resolve().parents[3] / "data" / "noise"


@dataclass
class Example:
    """One synthetic training example: audio, its ground-truth label, and generation params."""

    audio: np.ndarray
    sample_rate: int
    label: str
    wpm: float
    snr_db: float
    tone_freq_hz: float


def _random_text(rng: np.random.Generator, length_range: tuple[int, int]) -> str:
    length = int(rng.integers(length_range[0], length_range[1] + 1))
    return "".join(rng.choice(list(ALPHABET)) for _ in range(length))


def generate_example(
    text: str | None = None,
    *,
    wpm: float | None = None,
    snr_db: float | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    noise_dir: Path | str | None = DEFAULT_NOISE_DIR,
    wpm_range: tuple[float, float] = (15.0, 35.0),
    snr_range_db: tuple[float, float] = (-5.0, 15.0),
    freq_range_hz: tuple[float, float] = (400.0, 1000.0),
    length_range: tuple[int, int] = (10, 20),
    humanize: bool = True,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> Example:
    """Generate one synthetic noisy CW example.

    If `text`/`wpm`/`snr_db` are omitted they're drawn randomly from the given
    ranges. Set `humanize=False` to render "clean" ideal-timing CW with no
    jitter/drift/truncation and no QSB fading -- useful for sanity checks and
    for easy/curriculum examples.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    if text is None:
        text = _random_text(rng, length_range)
    if wpm is None:
        wpm = float(rng.uniform(*wpm_range))
    if snr_db is None:
        snr_db = float(rng.uniform(*snr_range_db))
    tone_freq = float(rng.uniform(*freq_range_hz))

    if humanize:
        events = key_text(
            text,
            wpm,
            wpm_drift_std=0.03,
            duration_jitter_std=0.12,
            space_truncation_prob=0.08,
            leading_silence_s=float(rng.uniform(0.1, 0.5)),
            trailing_silence_s=float(rng.uniform(0.1, 0.5)),
            rng=rng,
        )
        tone = key_to_audio(
            events,
            sample_rate,
            freq_hz=tone_freq,
            freq_drift_hz=0.5,
            amplitude_jitter_std=0.1,
            rng=rng,
        )
        tone *= qsb_envelope(len(tone), sample_rate, rng, depth=0.5)
    else:
        events = key_text(text, wpm, leading_silence_s=0.2, trailing_silence_s=0.2, rng=rng)
        tone = key_to_audio(events, sample_rate, freq_hz=tone_freq, rng=rng)

    noise = _build_noise(len(tone), sample_rate, noise_dir, rng)
    audio = mix_at_snr(tone, noise, snr_db)

    return Example(
        audio=audio,
        sample_rate=sample_rate,
        label=text,
        wpm=wpm,
        snr_db=snr_db,
        tone_freq_hz=tone_freq,
    )


def _build_noise(
    n_samples: int,
    sample_rate: int,
    noise_dir: Path | str | None,
    rng: np.random.Generator,
) -> np.ndarray:
    """A blend of synthetic broadband noise and (if available) a real environmental bed."""
    noise = 0.6 * white_noise(n_samples, rng) + 0.4 * pink_noise(n_samples, rng)

    if noise_dir is not None:
        bank = load_noise_bank(noise_dir, sample_rate)
        if any(bank.values()):
            try:
                bed = sample_noise_bed(bank, n_samples, sample_rate, rng)
                noise = 0.5 * noise + 0.5 * bed
            except ValueError:
                pass  # fall back to synthetic-only noise if the bank is empty/unreadable

    return noise.astype(np.float32)

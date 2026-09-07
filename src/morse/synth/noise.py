"""Noise generation and SNR-controlled mixing for synthetic CW training data.

Combines three kinds of noise, matching the target Raspberry Pi + cheap-mic
deployment scenario:
  - Real environmental recordings (`data/noise/`: anthropogenic, musical,
    voice, and wildlife beds, repurposed from the old presentation demo audio).
  - Synthetic broadband noise (white/pink).
  - QSB (fading): a slow amplitude envelope applied to the *tone*, simulating
    ionospheric fading that makes threshold-based decoders unreliable.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import lfilter

# Categories under data/noise/, each a distinct kind of environmental noise bed.
NOISE_CATEGORIES = ("anthro", "chords", "notes", "voice", "wildlife")

# Paul Kellett's "economy" pink-noise filter coefficients (a standard,
# widely-used approximation of a -3dB/octave IIR filter applied to white noise).
_PINK_B = np.array([0.049922035, -0.095993537, 0.050612699, -0.004408786])
_PINK_A = np.array([1.0, -2.494956002, 2.017265875, -0.522189400])


def white_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Unit-variance Gaussian white noise."""
    return rng.normal(0.0, 1.0, n_samples).astype(np.float32)


def pink_noise(n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """Approximately 1/f ("pink") noise, filtered from white noise."""
    white = rng.normal(0.0, 1.0, n_samples).astype(np.float64)
    pink = lfilter(_PINK_B, _PINK_A, white)
    peak = np.max(np.abs(pink))
    if peak > 0:
        pink = pink / peak
    return pink.astype(np.float32)


def qsb_envelope(
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
    *,
    min_period_s: float = 2.0,
    max_period_s: float = 8.0,
    depth: float = 0.6,
) -> np.ndarray:
    """A slow sinusoidal fading envelope in [1 - depth, 1], simulating QSB.

    depth=0 gives a constant envelope of 1 (no fading); depth=1 allows the
    signal to fade all the way to 0 at the trough.
    """
    depth = float(np.clip(depth, 0.0, 1.0))
    period_s = rng.uniform(min_period_s, max_period_s)
    phase = rng.uniform(0.0, 2 * np.pi)
    t = np.arange(n_samples) / sample_rate
    lfo = 0.5 * (1 + np.sin(2 * np.pi * t / period_s + phase))  # in [0, 1]
    return (1.0 - depth * (1.0 - lfo)).astype(np.float32)


def load_noise_bank(noise_dir: Path | str, sample_rate: int) -> dict[str, list[Path]]:
    """List available noise files per category under `noise_dir`, without loading audio yet.

    Actual audio is loaded lazily (per-draw) by `sample_noise_bed` since the
    bank can be large; this just indexes what's available.
    """
    noise_dir = Path(noise_dir)
    bank: dict[str, list[Path]] = {}
    for category in NOISE_CATEGORIES:
        category_dir = noise_dir / category
        if category_dir.is_dir():
            bank[category] = sorted(category_dir.glob("*.wav"))
    return bank


def sample_noise_bed(
    bank: dict[str, list[Path]],
    n_samples: int,
    sample_rate: int,
    rng: np.random.Generator,
    category: str | None = None,
) -> np.ndarray:
    """Draw `n_samples` of audio from a random (or specified-category) noise bed.

    Loops the source clip if it's shorter than `n_samples`, and picks a random
    start offset if it's longer. Requires librosa (resamples to `sample_rate`).
    """
    import librosa  # local import: keeps this module's import cost low when unused

    available = [c for c in (bank if category is None else {category: bank.get(category, [])}) if bank.get(c)]
    if not available:
        raise ValueError(f"no noise files available (category={category!r})")
    chosen_category = rng.choice(available)
    path = rng.choice(bank[chosen_category])

    audio, _ = librosa.load(str(path), sr=sample_rate, mono=True)
    if len(audio) == 0:
        raise ValueError(f"noise file {path} loaded as empty")

    if len(audio) < n_samples:
        repeats = n_samples // len(audio) + 1
        audio = np.tile(audio, repeats)
    start = rng.integers(0, len(audio) - n_samples + 1)
    return audio[start:start + n_samples].astype(np.float32)


def measured_snr_db(signal: np.ndarray, noise: np.ndarray) -> float:
    """Signal-to-noise ratio, in dB, of `signal` against `noise` (same length)."""
    signal_power = np.mean(signal.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)
    if noise_power <= 0:
        return float("inf")
    return 10 * np.log10(signal_power / noise_power)


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    *,
    eps: float = 1e-12,
) -> np.ndarray:
    """Scale `noise` so the mix hits the requested SNR (dB) against `signal`, then add.

    `signal` and `noise` must be the same length.
    """
    if len(signal) != len(noise):
        raise ValueError(f"signal and noise must be the same length, got {len(signal)} vs {len(noise)}")

    signal_power = np.mean(signal.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)
    target_noise_power = signal_power / (10 ** (snr_db / 10))
    scale = np.sqrt(target_noise_power / (noise_power + eps))
    return (signal + noise * scale).astype(np.float32)

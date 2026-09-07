import numpy as np
import pytest

from morse.realtime.segmenter import MessageSegmenter, SegmenterConfig


def _tone(duration_s: float, sample_rate: int, amplitude: float = 0.5) -> np.ndarray:
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 600 * t)).astype(np.float32)


def _silence(duration_s: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(duration_s * sample_rate), dtype=np.float32)


def _config(**overrides) -> SegmenterConfig:
    defaults = dict(sample_rate=8000, frame_ms=20.0, silence_rms_threshold=0.05, min_silence_s=0.5, min_active_s=0.05)
    defaults.update(overrides)
    return SegmenterConfig(**defaults)


def test_no_segment_finalized_during_pure_silence():
    config = _config()
    segmenter = MessageSegmenter(config)
    finalized = segmenter.push(_silence(2.0, config.sample_rate))
    assert finalized == []


def test_no_segment_finalized_while_still_active():
    config = _config()
    segmenter = MessageSegmenter(config)
    finalized = segmenter.push(_tone(1.0, config.sample_rate))
    assert finalized == []  # no closing silence gap yet


def test_segment_finalized_after_silence_gap():
    config = _config()
    segmenter = MessageSegmenter(config)
    chunk = np.concatenate([
        _tone(1.0, config.sample_rate),
        _silence(config.min_silence_s + 0.1, config.sample_rate),
    ])
    finalized = segmenter.push(chunk)
    assert len(finalized) == 1
    # The segment finalizes as soon as the silence run hits min_silence_s -- it
    # doesn't keep absorbing whatever extra silence happens to follow in the chunk.
    assert len(finalized[0]) / config.sample_rate == pytest.approx(1.0 + config.min_silence_s, abs=0.05)


def test_multiple_messages_in_one_stream():
    config = _config()
    segmenter = MessageSegmenter(config)
    chunk = np.concatenate([
        _tone(0.5, config.sample_rate),
        _silence(config.min_silence_s + 0.1, config.sample_rate),
        _tone(0.3, config.sample_rate),
        _silence(config.min_silence_s + 0.1, config.sample_rate),
    ])
    finalized = segmenter.push(chunk)
    assert len(finalized) == 2


def test_short_blip_is_discarded_as_noise():
    config = _config(min_active_s=0.2)
    segmenter = MessageSegmenter(config)
    chunk = np.concatenate([
        _tone(0.05, config.sample_rate),  # shorter than min_active_s
        _silence(config.min_silence_s + 0.1, config.sample_rate),
    ])
    finalized = segmenter.push(chunk)
    assert finalized == []


def test_max_message_s_forces_finalization():
    config = _config(max_message_s=1.0, min_silence_s=100.0)  # silence gap essentially disabled
    segmenter = MessageSegmenter(config)
    # Continuous tone well past max_message_s, no silence at all.
    finalized = segmenter.push(_tone(2.5, config.sample_rate))
    assert len(finalized) >= 1
    for segment in finalized:
        assert len(segment) / config.sample_rate <= config.max_message_s + 1e-6


def test_flush_returns_buffered_active_segment():
    config = _config()
    segmenter = MessageSegmenter(config)
    segmenter.push(_tone(0.5, config.sample_rate))  # no closing silence yet
    flushed = segmenter.flush()
    assert flushed is not None
    assert len(flushed) / config.sample_rate == pytest.approx(0.5, abs=0.05)


def test_flush_with_nothing_buffered_returns_none():
    config = _config()
    segmenter = MessageSegmenter(config)
    assert segmenter.flush() is None


def test_pushing_across_multiple_calls_still_finalizes():
    # The silence gap can straddle multiple push() calls (as real streaming would).
    config = _config()
    segmenter = MessageSegmenter(config)
    assert segmenter.push(_tone(0.5, config.sample_rate)) == []
    half_silence = config.min_silence_s / 2 + 0.05
    assert segmenter.push(_silence(half_silence, config.sample_rate)) == []
    finalized = segmenter.push(_silence(half_silence + 0.1, config.sample_rate))
    assert len(finalized) == 1

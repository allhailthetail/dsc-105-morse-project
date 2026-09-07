"""Buffer raw audio and detect message boundaries via a silence gap (simple energy-based VAD).

Deliberately does NOT run any model inference -- it only tracks short-time
RMS energy (via `morse.features.amplitude.rms_amplitude`, already dependency-
free) to decide when a burst of keyed activity has ended. Inference is
comparatively expensive (the exported model has a fixed, ~20s frame budget --
see `morse.realtime.config`), so it should run once per finalized message,
not on every incoming audio chunk. This module is pure numpy and has no
TensorFlow/TFLite dependency, so it's unit-testable without any ML runtime.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from morse.features.amplitude import rms_amplitude
from morse.realtime.config import MAX_MESSAGE_SECONDS


@dataclass
class SegmenterConfig:
    sample_rate: int = 16_000
    frame_ms: float = 20.0
    # NOTE: silence_rms_threshold, min_silence_s, and min_active_s are all
    # heuristic defaults, not calibrated against a real microphone/gain
    # setup -- expect to need tuning once running against actual hardware
    # (see scripts/run_realtime.py's corresponding CLI flags).
    silence_rms_threshold: float = 0.02
    min_silence_s: float = 1.0   # silence duration that finalizes a message
    min_active_s: float = 0.1    # discard segments shorter than this (likely noise blips)
    max_message_s: float = MAX_MESSAGE_SECONDS  # hard cap; must match the exported model's fixed frame budget


class MessageSegmenter:
    """Feed raw audio chunks in; get back finalized message segments as they complete."""

    def __init__(self, config: SegmenterConfig | None = None):
        self.config = config or SegmenterConfig()
        self._frame_len = max(1, int(self.config.frame_ms / 1000 * self.config.sample_rate))
        self._buffer: list[np.ndarray] = []
        self._active = False
        self._silence_run_s = 0.0
        self._active_run_s = 0.0  # non-silent time only -- distinct from total buffered duration

    def _buffered_seconds(self) -> float:
        return sum(len(f) for f in self._buffer) / self.config.sample_rate

    def _finalize(self) -> np.ndarray | None:
        segment = np.concatenate(self._buffer) if self._buffer else None
        active_run_s = self._active_run_s
        self._buffer = []
        self._active = False
        self._silence_run_s = 0.0
        self._active_run_s = 0.0
        # Discard by actual keyed-activity duration, not total segment length
        # (the segment also includes trailing silence collected while waiting
        # for the gap to close, which would otherwise make this check useless).
        if segment is not None and active_run_s < self.config.min_active_s:
            return None  # too short -- likely a noise blip, discard
        return segment

    def push(self, chunk: np.ndarray) -> list[np.ndarray]:
        """Feed one chunk of raw audio; returns zero or more finalized message segments."""
        finalized: list[np.ndarray] = []

        for start in range(0, len(chunk), self._frame_len):
            frame = chunk[start:start + self._frame_len]
            if len(frame) == 0:
                continue

            is_silent = rms_amplitude(frame) < self.config.silence_rms_threshold

            if not is_silent:
                self._buffer.append(frame)
                self._active = True
                self._silence_run_s = 0.0
                self._active_run_s += len(frame) / self.config.sample_rate
            elif self._active:
                self._buffer.append(frame)  # keep some trailing silence in the segment
                self._silence_run_s += len(frame) / self.config.sample_rate
                if self._silence_run_s >= self.config.min_silence_s:
                    segment = self._finalize()
                    if segment is not None:
                        finalized.append(segment)
                    continue
            # else: silence with nothing buffered yet -- ignore

            if self._buffered_seconds() >= self.config.max_message_s:
                segment = self._finalize()
                if segment is not None:
                    finalized.append(segment)

        return finalized

    def flush(self) -> np.ndarray | None:
        """Force-finalize whatever's currently buffered (e.g. on shutdown/EOF)."""
        return self._finalize() if self._buffer else None

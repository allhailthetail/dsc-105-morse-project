"""Shared constants for real-time inference.

`MAX_MESSAGE_SECONDS` bounds both the message segmenter's hard cutoff
(`morse.realtime.segmenter.SegmenterConfig.max_message_s`) and the fixed time
axis a TFLite-exported model is built for (`scripts/export_tflite.py`) --
these two must stay consistent, since a fixed-shape exported model can't
accept a segment longer than what it was built for.

Chosen with margin above the synthetic generator's own defaults: at its
slowest default WPM (15) and longest default message length (20 chars --
`morse.synth.generator.generate_example`'s `length_range`), the WPM/duration
relationship (`morse.timing.approx_wpm`'s inverse) puts a message at roughly
12*20/15 ~= 16s, plus up to ~1s of leading/trailing silence padding -- 20s
leaves comfortable headroom without making the fixed-shape model needlessly
large.
"""
from __future__ import annotations

from morse.features.melspec import HOP_LENGTH, expected_num_frames
from morse.synth.generator import DEFAULT_SAMPLE_RATE

MAX_MESSAGE_SECONDS = 20.0
MAX_MESSAGE_FRAMES = expected_num_frames(int(MAX_MESSAGE_SECONDS * DEFAULT_SAMPLE_RATE), HOP_LENGTH)

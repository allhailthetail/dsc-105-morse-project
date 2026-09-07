#!/usr/bin/env python
"""Real-time console CW decoder: mic (or a wav file) -> segment -> TFLite -> console.

Buffers incoming audio and watches for a silence gap (a simple energy-based
VAD, morse.realtime.segmenter.MessageSegmenter) to detect when a
transmission has ended; only then does it run the (comparatively expensive)
model inference and print a decoded line -- there is no continuous
partial-hypothesis output.

Runs against a live microphone by default, or loops a wav file via
--input-wav (handy for testing this plumbing on a desktop with a real or
synthetic recording, without any audio hardware).

Usage:
    python scripts/run_realtime.py --model model.tflite --input-wav data/real/cw001.wav
    python scripts/run_realtime.py --model model.tflite                     # live mic
    python scripts/run_realtime.py --model model.tflite --list-devices
"""
from __future__ import annotations

import argparse
import queue
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from morse.realtime.config import MAX_MESSAGE_FRAMES
from morse.realtime.decode import decode_segment, load_interpreter
from morse.realtime.segmenter import MessageSegmenter, SegmenterConfig


def _print_message(text: str) -> None:
    timestamp = datetime.now().isoformat(timespec="seconds")
    print(f"[{timestamp}] {text}", flush=True)


def _handle_finalized_segments(segments: list[np.ndarray], sample_rate: int, interpreter) -> None:
    for segment in segments:
        text = decode_segment(segment, sample_rate, interpreter, max_frames=MAX_MESSAGE_FRAMES)
        if text:
            _print_message(text)


def run_from_wav(path: Path, sample_rate: int, segmenter: MessageSegmenter, interpreter, chunk_len: int) -> None:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != sample_rate:
        import librosa

        audio = librosa.resample(audio, orig_sr=sr, target_sr=sample_rate)

    for start in range(0, len(audio), chunk_len):
        chunk = audio[start:start + chunk_len]
        _handle_finalized_segments(segmenter.push(chunk), sample_rate, interpreter)

    final_segment = segmenter.flush()
    if final_segment is not None:
        _handle_finalized_segments([final_segment], sample_rate, interpreter)


def run_from_microphone(
    sample_rate: int, segmenter: MessageSegmenter, interpreter, chunk_len: int, device: int | None
) -> None:
    import sounddevice as sd

    chunk_queue: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(f"warning: {status}", file=sys.stderr)
        chunk_queue.put(indata[:, 0].copy())  # keep the audio callback itself lightweight

    with sd.InputStream(
        channels=1, samplerate=sample_rate, blocksize=chunk_len, device=device, callback=callback
    ):
        print("listening for CW... (Ctrl+C to stop)", flush=True)
        try:
            while True:
                chunk = chunk_queue.get()
                _handle_finalized_segments(segmenter.push(chunk), sample_rate, interpreter)
        except KeyboardInterrupt:
            final_segment = segmenter.flush()
            if final_segment is not None:
                _handle_finalized_segments([final_segment], sample_rate, interpreter)
            print("\nstopped.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=Path, required=False, help="path to a .tflite model (see scripts/export_tflite.py)")
    parser.add_argument("--input-wav", type=Path, default=None, help="loop a wav file instead of a live mic")
    parser.add_argument("--device", type=int, default=None, help="sounddevice input device index")
    parser.add_argument("--list-devices", action="store_true", help="print available audio input devices and exit")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--chunk-ms", type=float, default=250.0, help="how often raw audio is checked for silence")
    parser.add_argument("--silence-threshold", type=float, default=SegmenterConfig().silence_rms_threshold)
    parser.add_argument("--min-silence-s", type=float, default=SegmenterConfig().min_silence_s)
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return

    if args.model is None:
        parser.error("--model is required unless --list-devices is given")

    config = SegmenterConfig(
        sample_rate=args.sample_rate,
        silence_rms_threshold=args.silence_threshold,
        min_silence_s=args.min_silence_s,
    )
    segmenter = MessageSegmenter(config)
    interpreter = load_interpreter(str(args.model))
    chunk_len = int(args.chunk_ms / 1000 * args.sample_rate)

    if args.input_wav is not None:
        run_from_wav(args.input_wav, args.sample_rate, segmenter, interpreter, chunk_len)
    else:
        run_from_microphone(args.sample_rate, segmenter, interpreter, chunk_len, args.device)


if __name__ == "__main__":
    main()

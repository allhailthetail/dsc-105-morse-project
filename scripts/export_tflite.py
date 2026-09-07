#!/usr/bin/env python
"""Export a trained checkpoint to TensorFlow Lite for Raspberry Pi4 inference.

Builds a **fixed-shape** clone of the model (concrete time axis, batch size
1) for conversion rather than converting the dynamic-shape training model
directly -- a dynamic time axis through a Bidirectional LSTM does not
convert with TFLite's builtin ops (confirmed empirically: it requires the
Flex-ops delegate, which defeats the point of the lightweight
`tflite-runtime` package on a Pi4). The fixed shape must match
`morse.realtime.config.MAX_MESSAGE_FRAMES` -- `scripts/run_realtime.py` pads
every captured segment's mel-spectrogram up to that many frames before
inference, so the two stay in sync automatically as long as neither hardcodes
a different number.

Usage:
    python scripts/export_tflite.py --checkpoint checkpoints/best.weights.h5 --out model.tflite
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from morse.features.melspec import N_MELS
from morse.models.crnn import build_crnn
from morse.realtime.config import MAX_MESSAGE_FRAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("model.tflite"))
    parser.add_argument(
        "--time-steps", type=int, default=MAX_MESSAGE_FRAMES,
        help="fixed frame count for the exported model (default: morse.realtime.config.MAX_MESSAGE_FRAMES)",
    )
    args = parser.parse_args()

    # Load the checkpoint into the normal dynamic-shape model (what it was trained/saved as)...
    trained_model = build_crnn(n_mels=N_MELS)
    trained_model.load_weights(str(args.checkpoint))

    # ...then copy weights into a fixed-shape clone for conversion. Weights are
    # shape-independent (Conv2D/LSTM parameters don't depend on the time axis).
    export_model = build_crnn(n_mels=N_MELS, time_steps=args.time_steps, batch_size=1)
    export_model.set_weights(trained_model.get_weights())

    converter = tf.lite.TFLiteConverter.from_keras_model(export_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range quantization
    try:
        tflite_model = converter.convert()
    except Exception as exc:  # noqa: BLE001 -- report clearly, then retry with a heavier fallback
        print(f"builtin-ops-only conversion failed ({exc}); retrying with SELECT_TF_OPS fallback...")
        print("NOTE: a model that needs this fallback requires the Flex delegate to run --")
        print("the lightweight tflite-runtime package alone will NOT work; use full tensorflow instead.")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS,
            tf.lite.OpsSet.SELECT_TF_OPS,
        ]
        tflite_model = converter.convert()

    args.out.write_bytes(tflite_model)
    print(f"wrote {args.out} ({len(tflite_model)} bytes, fixed at {args.time_steps} frames x {N_MELS} mels)")


if __name__ == "__main__":
    main()

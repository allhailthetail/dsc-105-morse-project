#!/usr/bin/env python
"""Train the CRNN+CTC model on streaming synthetic CW data.

Training data is an infinite stream (morse.data.pipeline.make_dataset), so an
"epoch" here means --steps-per-epoch batches, not one pass over a fixed set.
Validation uses a small, fixed (seed-1) in-memory synthetic set, generated
once at startup -- NOT the same as data/real/. This script only checks "did
it learn the synthetic distribution"; run scripts/evaluate.py afterwards for
the real-world (data/real/) numbers that actually matter.

Usage:
    python scripts/train.py --epochs 2 --steps-per-epoch 20   # tiny smoke-test run
    python scripts/train.py --epochs 50 --steps-per-epoch 500 --mixed-precision
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

# Must run before any GPU is initialized (raises RuntimeError if called
# late, e.g. after the first GPU op) -- TF otherwise reserves nearly all
# device memory as one upfront arena, which under WSL2's GPU passthrough
# has been implicated in "Unexpected Event status" CUDA-runtime aborts when
# a training step first touches a much-larger-than-previously-seen tensor.
for _gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(_gpu, True)

from morse.data.pipeline import make_dataset, prepare_for_training
from morse.features.melspec import N_MELS, log_mel_spectrogram
from morse.labels.vocab import encode
from morse.models.crnn import build_crnn, ctc_loss
from morse.synth.generator import DEFAULT_SAMPLE_RATE, generate_example

VAL_SEED = 1  # matches the val-set seed convention in docs/HOWTO-generate-corpus.md


def build_fixed_validation_set(
    n: int, seed: int, sample_rate: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """A small, fixed, padded (features, labels, input_length, label_length) validation batch.

    Generated once in-memory (not written to disk) so this script is
    self-contained -- doesn't depend on scripts/generate_dataset.py having
    been run first.
    """
    rng = np.random.default_rng(seed)
    feats, labels = [], []
    for _ in range(n):
        example = generate_example(sample_rate=sample_rate, rng=rng)
        feats.append(log_mel_spectrogram(example.audio, sample_rate))
        labels.append(encode(example.label))

    input_length = np.array([f.shape[0] for f in feats], dtype=np.int32)
    label_length = np.array([l.shape[0] for l in labels], dtype=np.int32)
    max_t, max_l = int(input_length.max()), int(label_length.max())

    features = np.zeros((n, max_t, N_MELS), dtype=np.float32)
    label_ids = np.zeros((n, max_l), dtype=np.int32)
    for i, (f, l) in enumerate(zip(feats, labels)):
        features[i, : f.shape[0]] = f
        label_ids[i, : l.shape[0]] = l

    return features, label_ids, input_length, label_length


# Batches have a different padded time/label length every step (padded_batch
# pads to each batch's own max, not a fixed size), so an undecorated
# `@tf.function` retraces on nearly every call -- extremely expensive (this
# was measured: ~240s for a 20-step epoch on a tiny model, almost entirely
# retracing overhead, with TF's graph optimizer logging repeated
# "tensor<NxMxK> not compatible" warnings along the way). An explicit
# `input_signature` with dynamic (`None`) shapes for the padded axes makes
# TF trace once and reuse that graph for every batch shape.
_STEP_SIGNATURE = [
    tf.TensorSpec(shape=(None, None, N_MELS), dtype=tf.float32),  # features
    tf.TensorSpec(shape=(None, None), dtype=tf.int32),            # labels
    tf.TensorSpec(shape=(None,), dtype=tf.int32),                 # input_length
    tf.TensorSpec(shape=(None,), dtype=tf.int32),                 # label_length
]


def make_train_step(model, optimizer, clip_norm: float):
    @tf.function(input_signature=_STEP_SIGNATURE)
    def train_step(features, labels, input_length, label_length):
        with tf.GradientTape() as tape:
            logits = model(features, training=True)
            loss = tf.reduce_mean(ctc_loss(labels, logits, label_length, input_length))
        grads = tape.gradient(loss, model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, clip_norm)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    return train_step


def make_val_step(model):
    @tf.function(input_signature=_STEP_SIGNATURE)
    def val_step(features, labels, input_length, label_length):
        logits = model(features, training=False)
        return tf.reduce_mean(ctc_loss(labels, logits, label_length, input_length))

    return val_step


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-size", type=int, default=200)
    parser.add_argument("--val-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0, help="training stream seed (see docs/HOWTO-generate-corpus.md)")
    parser.add_argument("--mixed-precision", action="store_true", help="opt-in; mainly benefits GPU training")
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="widen WPM/SNR ranges over training; off by default (train on the as-is synthetic distribution)",
    )
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()

    if args.mixed_precision:
        import keras

        keras.mixed_precision.set_global_policy("mixed_float16")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    print(f"building fixed validation set ({args.val_size} examples, seed={VAL_SEED})...")
    val_features, val_labels, val_input_length, val_label_length = build_fixed_validation_set(
        args.val_size, VAL_SEED, DEFAULT_SAMPLE_RATE
    )

    model = build_crnn(n_mels=N_MELS)
    model.summary()
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    train_step = make_train_step(model, optimizer, args.grad_clip_norm)
    val_step = make_val_step(model)

    log_path = args.log_dir / "train.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])

    best_val_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()

        curriculum_kwargs = {}
        if args.curriculum:
            # Widen ranges linearly over training: start easy (high SNR, slow
            # WPM), end at generate_example's own full default ranges.
            progress = (epoch - 1) / max(args.epochs - 1, 1)
            curriculum_kwargs = {
                "wpm_range": (15.0, 15.0 + progress * 20.0),
                "snr_range_db": (15.0 - progress * 20.0, 15.0),
            }

        train_ds = prepare_for_training(
            make_dataset(seed=args.seed, **curriculum_kwargs), batch_size=args.batch_size
        )

        train_losses = []
        for step, (features, labels, input_length, label_length) in enumerate(train_ds.take(args.steps_per_epoch)):
            loss = train_step(features, labels, input_length, label_length)
            train_losses.append(float(loss))
            if step % 50 == 0:
                print(f"  epoch {epoch} step {step}/{args.steps_per_epoch} loss={float(loss):.3f}")

        val_losses = []
        for start_idx in range(0, args.val_size, args.val_batch_size):
            end_idx = start_idx + args.val_batch_size
            loss = val_step(
                val_features[start_idx:end_idx],
                val_labels[start_idx:end_idx],
                val_input_length[start_idx:end_idx],
                val_label_length[start_idx:end_idx],
            )
            val_losses.append(float(loss))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        elapsed = time.time() - start
        print(f"epoch {epoch}/{args.epochs}: train_loss={train_loss:.3f} val_loss={val_loss:.3f} ({elapsed:.1f}s)")

        with log_path.open("a", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow([epoch, train_loss, val_loss, round(elapsed, 1)])

        model.save_weights(str(args.checkpoint_dir / "last.weights.h5"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model.save_weights(str(args.checkpoint_dir / "best.weights.h5"))
            print(f"  new best val_loss={val_loss:.3f}, saved checkpoints/best.weights.h5")

    print(f"done. checkpoints in {args.checkpoint_dir}, log at {log_path}")


if __name__ == "__main__":
    main()

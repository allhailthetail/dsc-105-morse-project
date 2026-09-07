# HOW TO: Generate a Training Corpus

This walks through using `src/morse/` to generate synthetic CW (Morse code)
audio for training, from a single sanity-checked example up to a full corpus.

## Before you start

There are two separate paths, depending on what the data is for:

| | Fixed corpus on disk | Live streaming |
|---|---|---|
| Tool | `scripts/generate_dataset.py` | `morse.data.pipeline.make_dataset()` |
| Output | `.wav` files + `labels.csv` | `(mel_spectrogram, label_ids)` tensors, generated on the fly |
| Use for | validation/test sets (need to be fixed & reproducible) | training (wants effectively unlimited variety) |
| Needs TensorFlow? | No | Yes (`requirements-ml.txt`) |
| Cost | ~620KB/example on disk, ~26 examples/sec on one core | no disk cost, same ~26/sec generation rate but consumed live |

**Rule of thumb**: generation runs at roughly 26 examples/second, single-threaded,
and each rendered example is about 620KB on disk (16kHz float32 wav, ~7-9s
average clip). So:

- A 5,000-example validation set is about 3 minutes and ~3GB.
- A 100,000-example *rendered* set is about an hour and **~60GB** — past this
  scale, stream training data instead of pre-rendering it (see Step 3).

## Step 1 — Activate the environment

```bash
cd old-project
source .venv/Scripts/activate    # or .venv/bin/activate on Linux/macOS
```

Core dependencies (`numpy`, `scipy`, `soundfile`, `librosa`, etc.) come from
`requirements.txt` and are all you need for Steps 1-2 and 4 below.
`requirements-ml.txt` (TensorFlow/Keras) is only needed for Step 3.

## Step 2 — Sanity-check one example before generating thousands

```bash
python scripts/preview_synth.py --text "CQ CQ DE W1AW" --wpm 20 --snr-db 5
```

This writes `plots/synth_preview/`:
- `preview.wav` — listen to it
- `waveform.png` — should show clear on/off keying bursts
- `melspec.png` — should show a horizontal tone band with visible dit/dash patterning

Adjust `--wpm`/`--snr-db`/`--text` (or omit them for random values) until the
output looks/sounds like what you expect before moving on.

## Step 3 — Render fixed validation/test sets to disk

```bash
python scripts/generate_dataset.py --n 5000 --seed 1 --out datasets/val
python scripts/generate_dataset.py --n 2000 --seed 2 --out datasets/test
```

Use a **different `--seed` per split** so val/test examples never collide with
each other or with the training seed used in Step 4. Each run writes:

```
datasets/val/
  synth00000.wav ... synth04999.wav
  labels.csv   # filename,transcription,wpm,snr_db,tone_freq_hz
```

**Known limitation**: the CLI only exposes `--n`/`--seed`/`--out` — WPM range,
SNR range, and tone-frequency range are hardcoded defaults inside
`generate_example()`. For a curriculum split (e.g. easy: high SNR/slow WPM vs.
hard: low SNR/fast WPM), call
`morse.synth.generator.generate_example(...)` directly with your own ranges in
a small custom script instead of the CLI.

## Step 4 — Stream training data (don't render it)

```bash
pip install -r requirements-ml.txt   # tensorflow + keras, only needed here
```

```python
import tensorflow as tf
from morse.data.pipeline import make_dataset

train_ds = make_dataset(seed=0)  # infinite generator; seed 0 avoids colliding with val/test above
train_ds = train_ds.padded_batch(
    batch_size=32,
    padding_values=(0.0, -1),  # pad audio with 0, labels with an ignorable sentinel
)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

for features, labels in train_ds.take(1):
    print(features.shape, labels.shape)  # (32, T, 40), (32, L)
```

Each batch draws fresh random messages/WPM/SNR/noise mixes — there's no fixed
size to run out of, and nothing touches disk.

**Gap to plan around**: CTC loss needs per-example *input length* and *label
length* tensors alongside the padded batch, so the loss knows how much of each
padded row is real vs. padding. `make_dataset()` doesn't emit those yet — that
plumbing belongs with the CRNN training code itself (see
[`src/morse/models/README.md`](../src/morse/models/README.md)), not the data
pipeline.

## Step 5 — Load a rendered set back (for eval scripts, external tools, etc.)

```python
import pandas as pd
import soundfile as sf
from morse.features.melspec import log_mel_spectrogram

labels = pd.read_csv("datasets/val/labels.csv")
for _, row in labels.iterrows():
    audio, sr = sf.read(f"datasets/val/{row.filename}")
    features = log_mel_spectrogram(audio, sr)
    # features: (T, 40), row.transcription: ground-truth string
```

## If you need a huge corpus on disk anyway

`scripts/generate_dataset.py` is currently single-threaded and serial —
rendering 500k+ examples would take several hours. Since Step 4's streaming
pipeline already covers training without this cost, only go this route for a
specific reason (e.g. handing data to a non-Python tool). Parallelizing the
script across cores would be the way to speed this up if/when it's needed.

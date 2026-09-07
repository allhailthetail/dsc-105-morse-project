# HOW TO: Train the CRNN Model (GPU / WSL)

This is the training-side counterpart to
[`HOWTO-generate-corpus.md`](HOWTO-generate-corpus.md). It walks through
running a real training run on a GPU, and how to tell whether the model is
actually learning.

## Why this needs to happen on your own machine, not here

This project was built and tested in a CPU-only sandbox — `tf.config.list_physical_devices('GPU')`
returns `[]` there. A real training run (as opposed to a correctness smoke
test) needs an actual GPU. `src/morse/data/pipeline.py`'s bucketed-batching
default is already tuned against an RTX 3090 (see that file's docstring), so
if that's your card, you're starting from a measured baseline, not a guess.

## Step 1 — One-time WSL2 + GPU setup (skip if already done)

- Make sure the **Windows host** has a current NVIDIA driver with WSL/CUDA
  support (a recent Game Ready or Studio driver — check nvidia.com). Don't
  install a separate NVIDIA driver *inside* WSL; WSL uses driver passthrough
  from Windows.
- Inside WSL Ubuntu, confirm the GPU is visible:
  ```bash
  nvidia-smi
  ```
  This must show your GPU before going any further. If it doesn't, fix
  driver passthrough first (NVIDIA's "CUDA on WSL" guide) — nothing below
  will work around a broken passthrough.

## Step 2 — Get the repo onto WSL's native filesystem

Don't work from `/mnt/c/...` — cross-filesystem I/O between Windows and WSL
is slow, and this project reads a lot of small files (`data/real/`,
`data/noise/`). Clone or copy the repo to somewhere under WSL's own
filesystem instead:

```bash
cd ~
git clone <your repo remote>  old-project   # or: cp -r /mnt/c/Users/.../old-project ~/old-project
cd old-project
git checkout python-migration   # or whichever branch has this work
```

## Step 3 — Python environment

```bash
python3 --version   # verify this has a tensorflow[and-cuda] wheel available -- check before installing (see note below)
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
pip install -r requirements-ml.txt   # tensorflow[and-cuda] + keras
```

**Note**: this repo's Windows dev sandbox used Python 3.13, but that was only
verified against the plain `tensorflow` wheel there — `tensorflow[and-cuda]`'s
Linux/GPU wheel support for a given Python version hasn't been checked from
this session at all. Run `pip index versions tensorflow` in your WSL venv
first; if 3.13 isn't covered, fall back to 3.11 or 3.12 (via `pyenv` or
`deadsnakes`) rather than fighting an unsupported combination.

## Step 4 — Confirm TF actually sees the GPU

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

Should print one `PhysicalDevice(..., device_type='GPU')` entry, not `[]`. If
it's empty here despite `nvidia-smi` working, that's a TF/CUDA/driver version
mismatch — check TensorFlow's install docs for the CUDA/driver versions your
installed `tensorflow[and-cuda]` release actually expects, rather than
training on CPU without noticing.

## Step 5 — Quick smoke test first (a few minutes, not a real run)

```bash
python scripts/train.py --epochs 3 --steps-per-epoch 100 --val-size 100
```

Purpose: confirm the whole pipeline runs on *your* hardware, and get a real
per-step timing number — check `logs/train.csv`'s `seconds` column divided by
`steps-per-epoch` for actual seconds/step on your GPU. The CPU-sandbox
numbers this was built with (~9s/step before bucketing, meaningfully less
after) don't transfer to your hardware — measure your own before committing
to a long run.

Also check that `val_loss` has already started dropping at this tiny scale —
see "Reading the metrics" below for what that should look like.

## Step 6 — the real training run

A reasonable starting point (see "Picking epochs/steps" below for the
reasoning, not just the numbers):

```bash
python scripts/train.py --epochs 50 --steps-per-epoch 500 --batch-size 64
```

Adjust `--batch-size` upward if your GPU has memory to spare — this is a
small model (order of ~1M parameters), so a 3090-class card likely has room
for a larger batch than the default 32, which improves GPU utilization.
Adjust `--steps-per-epoch`/`--epochs` based on the Step 5 timing and how much
wall-clock time you're willing to spend.

`--curriculum` (widens WPM/SNR ranges over training) is available but off by
default — try the plain run first for a clean baseline consistent with the
project's "train on the as-is synthetic distribution first" strategy; treat
curriculum as a follow-up experiment if the baseline's convergence looks slow
or stuck, not a default.

## Step 7 — check progress mid-run (optional but recommended)

`checkpoints/best.weights.h5` updates every epoch that improves val loss, so
in a second terminal (same venv) you can check real-world progress without
stopping training:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.weights.h5
```

Doing this after the Step 5 smoke test and again partway through Step 6 is
worth the two minutes it costs — catching a structurally broken run early
(CER stuck at 1.0) is a lot cheaper than discovering it after a multi-hour
training run finishes.

## Step 8 — after training: evaluate, export, test the realtime loop

```bash
python scripts/evaluate.py --checkpoint checkpoints/best.weights.h5
python scripts/export_tflite.py --checkpoint checkpoints/best.weights.h5 --out model.tflite
python scripts/run_realtime.py --model model.tflite --input-wav data/real/cw001.wav
```

If you run multiple training experiments, copy checkpoints out between runs
(`cp checkpoints/best.weights.h5 checkpoints/run1_best.weights.h5`) —
`scripts/train.py` always writes to the same `best.weights.h5`/`last.weights.h5`
names and will overwrite them on the next run.

---

## Picking epochs/steps

With streaming/infinite synthetic data, "epoch" here just means
`--steps-per-epoch` batches — there's no fixed dataset to run out of, so the
real dial is *total steps* (`epochs × steps_per_epoch`), and the right values
depend on your actual hardware speed, not a number picked in advance. The
process:

1. Run the Step 5 smoke test, note real seconds/step from `logs/train.csv`.
2. Decide a wall-clock budget you're willing to spend on a first real run
   (e.g. "an hour or two").
3. Back into `--steps-per-epoch`/`--epochs` from that budget and your
   measured seconds/step, rather than trusting the `--epochs 50
   --steps-per-epoch 500` suggestion above as anything more than a starting
   point for a first attempt.
4. Use the metrics below (not a fixed step count) to decide whether to
   extend training further: if `val_loss` and the synthetic CER (via
   `scripts/evaluate.py`) are still improving run-over-run, there's no
   inherent reason to stop — there's no overfitting risk from "running out
   of examples" the way there would be on a fixed dataset. If they've
   plateaued, more steps at the same settings won't help; that's a signal to
   change something (learning rate, model size, curriculum) rather than just
   running longer.

## Reading the metrics: is it actually learning?

Two different layers of signal — don't rely on the first one alone:

**1. CTC loss (`train_loss`/`val_loss` in `logs/train.csv`)** — watch the
*trend*, not the absolute number. CTC loss isn't independently interpretable
or comparable across setups, and it has a nonzero floor even for a
hypothetically perfect model (it integrates over alignments, not a simple
per-frame error). What to look for:
- Both `train_loss` and `val_loss` trending down over epochs.
- `val_loss` roughly tracking `train_loss`, not diverging upward while
  `train_loss` keeps falling (that pattern would flag overfitting to the
  fixed 200-example validation set, or a learning-rate problem).
- A loss that's gone flat isn't necessarily broken — cross-check against
  the CER numbers below before concluding anything from loss alone.

**2. CER / exact-match via `scripts/evaluate.py`** — this is the metric that
actually answers "does it work," and it reports two numbers side by side on
purpose:
- **Synthetic set (in-distribution)**: should improve fastest and go
  lowest, since it's the training distribution. Rough, non-guaranteed
  benchmarks for judging your own run (there's no prior baseline for this
  exact setup to compare against, so treat these as sanity thresholds, not
  targets): if CER is still ~1.0 with 0% exact match after the Step 6
  baseline run, something is structurally broken (architecture, loss
  wiring, or data) — worth debugging before running longer. A CER
  meaningfully below ~0.3-0.5 with some non-zero exact-match rate is a
  reasonable "this is actually learning" signal.
- **Real set (`data/real/`, 58 labeled recordings, out-of-distribution)**:
  this is the number that matters for the actual deployment goal, and it's
  expected to lag behind the synthetic number. Per the project's chosen
  strategy: use the **gap** between synthetic CER and real CER as the
  signal for whether `morse.synth`'s humanization/noise parameters need
  revisiting — a small, closing gap means the synthetic distribution
  reasonably resembles real hand-sent CW; a large, stuck gap means the
  model is fitting artifacts specific to the synthetic generator rather
  than generalizing, which more training won't fix by itself.

Run `evaluate.py` against checkpoints from a few different points in
training (not just the very end) so you're looking at a trend across both
metrics, not a single snapshot.

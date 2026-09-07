# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CRNN (CNN + Bi-LSTM + CTC loss) that decodes hand-sent Morse code (CW) audio, trained on synthetic data and targeting deployment on a Raspberry Pi 4 with a cheap microphone, printing decoded messages to a console in real time. The project was originally an Emacs Org-mode + R analysis of 200 real recordings; it was migrated to Python and re-scoped around synthetic data generation, because real hand-sent recordings alone are too small and too fixed to train a sequence model that needs to handle speed drift, truncated inter-character spacing, and fading (QSB) — see `src/morse/models/README.md` for the full rationale.

**Chosen sim-to-real strategy** (don't re-litigate without new evidence): train on the synthetic generator's data *as-is* — its humanization/noise parameters (jitter, drift, truncation probability, SNR range) are hand-picked, not calibrated against real CW timing statistics, because no tooling exists to measure those from `data/real/`. Run `scripts/evaluate.py` after training and use the resulting real-vs-synthetic CER gap as the signal for whether those parameters need revisiting — not intuition.

## Commands

```bash
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate on Linux/macOS
pip install -r requirements.txt      # core: numpy, scipy, pandas, soundfile, librosa, matplotlib, sounddevice, pytest
pip install -e .                     # installs the morse package in editable mode
pip install -r requirements-ml.txt   # needed for anything touching morse.data.pipeline / morse.models — not for DSP/generator work

pytest                                              # run the full test suite
pytest tests/synth/test_keyer.py -k test_word_gap   # a single test

python scripts/preview_synth.py --text "CQ CQ DE W1AW" --wpm 20 --snr-db 5   # generate + plot one synthetic example
python scripts/generate_dataset.py --n 200 --seed 0 --out datasets/val      # render a fixed val/test set
python scripts/analyze_real_data.py                                        # QC pass over data/real/

python scripts/train.py --epochs 50 --steps-per-epoch 500                  # train (add --curriculum, --mixed-precision as wanted)
python scripts/evaluate.py --checkpoint checkpoints/best.weights.h5        # CER on synthetic AND data/real/ side by side
python scripts/export_tflite.py --checkpoint checkpoints/best.weights.h5 --out model.tflite
python scripts/run_realtime.py --model model.tflite --input-wav data/real/cw001.wav   # test without a live mic
python scripts/run_realtime.py --model model.tflite                                    # live mic, prints to console
```

## Directory structure

- `src/morse/` — the importable package.
  - `morse_code.py`, `timing.py` — the ITU Morse table and PARIS-standard timing (`tau = 6/(5*wpm)`; dot=tau, dash=3tau, gaps=1/3/7 tau).
  - `synth/` — the synthetic generator: `keyer.py` (text → humanized timing `Event` sequence: WPM drift, duration jitter, inter-character space truncation), `tone.py` (timing → keyed sine wave with envelope shaping + frequency drift), `noise.py` (white/pink noise, QSB fading envelope, SNR-controlled mixing with real noise beds), `generator.py` (ties it together: `generate_example()` is the main entry point).
  - `features/melspec.py` — **the single source of truth for mel-spectrogram parameters.** Training, evaluation, and the real-time script all call this unchanged — a mismatch here silently breaks the model. `features/metadata.py`/`amplitude.py` are ported DSP utilities (RMS, ZCR, per-file metadata), used mainly against `data/real/`.
  - `filters/bandpass.py`, `filters/anf.py` — Butterworth bandpass and auto-notch (auto center-frequency detection) filters, kept as optional preprocessing/robustness tools.
  - `io/wav.py` — WAV read/write. `read_wav` derives bit depth from the file's actual header rather than trusting stale metadata (see "Gotchas" below).
  - `labels/vocab.py` — character vocabulary (36 alphanumeric + **1 CTC blank at the *last* index**, not index 0 — see Gotchas) and text↔id encode/decode, including greedy CTC-output collapsing (`ctc_collapse`, pure numpy/python, no TF dependency).
  - `data/pipeline.py` — requires TensorFlow. Streams `(features, label_ids, input_length, label_length)` on the fly via `tf.data.Dataset.from_generator`, wrapping the synthetic generator — training data is effectively infinite. `prepare_for_training()` pads/batches it. Supports `wpm_range`/`snr_range_db`/etc. overrides for curriculum training.
  - `models/crnn.py` — the CRNN (`build_crnn`) plus `ctc_loss()`/`greedy_decode()` helpers built on `keras.ops.ctc_loss`/`ctc_decode` (not the high-level `keras.losses.CTC` class — see Gotchas). `models/README.md` documents the design.
  - `eval/metrics.py` — dependency-free character error rate (Levenshtein-based) and exact-match rate.
  - `realtime/` — the real-time console decoder's logic, deliberately TF-free where possible: `segmenter.py` (pure numpy energy-based VAD that buffers audio and finalizes a "message" on a silence gap — no model inference happens until a message is finalized, since inference is comparatively expensive), `decode.py` (loads a TFLite interpreter — `tflite_runtime` if present, else falls back to `tensorflow.lite` — pads/trims a segment's mel-spectrogram to the model's fixed frame budget, runs inference, decodes via `labels.vocab.ctc_collapse`), `config.py` (the shared `MAX_MESSAGE_SECONDS`/`MAX_MESSAGE_FRAMES` constants that keep the segmenter's cutoff and the exported model's fixed shape in sync).
- `scripts/` — entry points (not imported as a library): `generate_dataset.py`, `preview_synth.py`, `analyze_real_data.py`, `train.py`, `evaluate.py`, `export_tflite.py`, `run_realtime.py`.
- `tests/` — pytest, mirrors `src/morse/` layout. Tests touching TensorFlow use `pytest.importorskip("tensorflow")` so the rest of the suite runs without the ML stack installed.
- `data/real/` — the original 200 hand-sent CW recordings (`cw001.wav`–`cw200.wav`), archived as a **held-out real-world eval set** — never used for training. `data/real/labels.csv` holds the 58 known ground-truth transcriptions (extracted from the old `main.org`; the other 142 files have no known label).
- `data/noise/` — real environmental noise beds (`anthro/`, `chords/`, `notes/`, `voice/`, `wildlife/`), repurposed from the old presentation demo audio, used by `synth/noise.py` for SNR-controlled mixing.
- `legacy/archive/` — the original Org-mode/R/LaTeX implementation (`main.org`, `functions.org`, `docs/`), preserved for history. See its own README for what's there.
- `docs/future-work.md` — a lightweight web UI, persistent message logging, and driving the Pi's TFT display are **explicitly not built** — design notes only.

## Gotchas / non-obvious things

- **`data/real/*.wav` are 32-bit PCM, not 8-bit.** The old R analysis (`tuneR`) mis-reported bit depth; direct header inspection confirms 32-bit. `morse.io.wav.read_wav` derives dtype from the actual file header — don't hardcode assumptions from the archived R output.
- **Only 58 of the 200 real recordings have known labels** (`data/real/labels.csv`), sourced from the Kaggle "morse-challenge" competition via the old `main.org`. The other 142 have no ground truth.
- **Synthetic audio spec is 16kHz mono float32** (`morse.synth.generator.DEFAULT_SAMPLE_RATE`), independent of the real dataset's 8kHz — `scripts/evaluate.py`/`run_realtime.py` resample real/live audio up to 16kHz before feeding the model, not the other way around.
- **`keras.losses.CTC` (the high-level Loss class) is NOT used, deliberately.** Confirmed by reading its source: its functional form hardcodes `mask_index=0` and derives `label_length`/`input_length` by assuming the *entire padded time axis* is valid for every example — it has no way to accept per-example lengths, so it silently corrupts the loss for a padded batch of variable-length clips. `morse.models.crnn` calls the lower-level `keras.ops.ctc_loss`/`ctc_decode` directly instead, with explicit lengths and `mask_index=BLANK_INDEX` (blank is the *last* vocab index here, not the library's default of 0).
- **The CRNN's conv stack never pools/strides over the time axis, only frequency.** This is relied on directly: `build_crnn`'s output time dimension always exactly equals its input time dimension, so `input_length` doubles as the CTC output length with no separate calculation. Don't add time-axis pooling without revisiting that assumption.
- **TFLite export uses a fixed-shape model clone, not the dynamic-shape training model.** Confirmed empirically: converting the dynamic time-axis model (needed for training/eval on variable-length audio) fails on builtin TFLite ops (`TensorListReserve` requires a static shape) and only succeeds via the heavier `SELECT_TF_OPS`/Flex-delegate fallback — which defeats the point of the lightweight `tflite-runtime` package on a Pi4. `scripts/export_tflite.py` instead builds a fixed-`time_steps` clone (`build_crnn(time_steps=..., batch_size=1)`), copies weights via `get_weights()`/`set_weights()`, and that converts cleanly with builtins only. The fixed shape must match `morse.realtime.config.MAX_MESSAGE_FRAMES`.
- **The real-time script decodes once per finalized message, not on a continuous sliding window.** Given the fixed-shape export above, running inference every ~250ms on a ~20s-frame-budget model would be wasteful; the segmenter's cheap RMS-based VAD gates when the (comparatively expensive) model inference actually runs.
- **`silence_rms_threshold`/`min_silence_s` in `realtime/segmenter.py` are uncalibrated heuristic defaults**, not tuned against a real microphone/gain setup — expect to adjust via `run_realtime.py`'s CLI flags once running against actual hardware.
- **`requirements.txt` vs `requirements-ml.txt` is a hard boundary**, not just organizational tidiness: dataset generation and DSP work must stay usable without TensorFlow installed. `morse.realtime.decode` specifically avoids Keras (uses pure-numpy `ctc_collapse`, not `models.crnn.greedy_decode`) so the eventual Pi4 deployment can run on `tflite-runtime` alone rather than full TensorFlow.

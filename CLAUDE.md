# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A synthetic-data pipeline for training a CRNN (CNN + Bi-LSTM + CTC loss) to decode hand-sent Morse code (CW) audio, targeting deployment on a Raspberry Pi 4 with a cheap microphone. The project was originally an Emacs Org-mode + R analysis of 200 real recordings; it has been migrated to Python and re-scoped around synthetic data generation, because real hand-sent recordings alone are too small and too fixed to train a sequence model that needs to handle speed drift, truncated inter-character spacing, and fading (QSB) — see `src/morse/models/README.md` for the full rationale and target architecture.

**Model/training code is not implemented yet** — this repo currently covers only the synthetic data generator, DSP utilities, and feature extraction that will feed that future model.

## Commands

```bash
python -m venv .venv && source .venv/Scripts/activate   # or .venv/bin/activate on Linux/macOS
pip install -r requirements.txt      # core: numpy, scipy, pandas, soundfile, librosa, matplotlib, pytest
pip install -e .                     # installs the morse package in editable mode
pip install -r requirements-ml.txt   # ONLY needed for src/morse/data/pipeline.py (tensorflow/keras) — not for anything else

pytest                                              # run the full test suite
pytest tests/synth/test_keyer.py -k test_word_gap   # a single test

python scripts/preview_synth.py --text "CQ CQ DE W1AW" --wpm 20 --snr-db 5   # generate + plot one example
python scripts/generate_dataset.py --n 200 --seed 0 --out datasets/val      # render a fixed val/test set
python scripts/analyze_real_data.py                                        # QC pass over data/real/
```

## Directory structure

- `src/morse/` — the importable package.
  - `morse_code.py`, `timing.py` — the ITU Morse table and PARIS-standard timing (`tau = 6/(5*wpm)`; dot=tau, dash=3tau, gaps=1/3/7 tau).
  - `synth/` — the synthetic generator: `keyer.py` (text → humanized timing `Event` sequence: WPM drift, duration jitter, inter-character space truncation), `tone.py` (timing → keyed sine wave with envelope shaping + frequency drift), `noise.py` (white/pink noise, QSB fading envelope, SNR-controlled mixing with real noise beds), `generator.py` (ties it together: `generate_example()` is the main entry point).
  - `features/melspec.py` — **the single source of truth for mel-spectrogram parameters.** Any future embedded/inference preprocessing must match these exactly, or training/deployment will silently diverge. `features/metadata.py`/`amplitude.py` are ported DSP utilities (RMS, ZCR, per-file metadata), used mainly against `data/real/`.
  - `filters/bandpass.py`, `filters/anf.py` — Butterworth bandpass and auto-notch (auto center-frequency detection) filters, kept as optional preprocessing/robustness tools.
  - `io/wav.py` — WAV read/write. `read_wav` derives bit depth from the file's actual header rather than trusting stale metadata (see "Gotcha" below).
  - `labels/vocab.py` — character vocabulary (36 alphanumeric + 1 CTC blank) and text↔id encode/decode, including greedy CTC-output collapsing.
  - `data/pipeline.py` — **the only module that requires TensorFlow.** Streams `(mel_spectrogram, label_ids)` pairs on the fly via `tf.data.Dataset.from_generator`, wrapping the synthetic generator — training data is effectively infinite, there's no fixed on-disk dataset.
  - `models/README.md` — documents the target CRNN+CTC architecture; no model code yet.
- `scripts/` — entry points (not imported as a library): `generate_dataset.py` (fixed-seed dataset rendering), `preview_synth.py` (generate + plot one example), `analyze_real_data.py` (metadata/RMS/ZCR QC over `data/real/`).
- `tests/` — pytest, mirrors `src/morse/` layout. `tests/data/test_pipeline.py` uses `pytest.importorskip("tensorflow")` so the rest of the suite runs without the ML stack installed.
- `data/real/` — the original 200 hand-sent CW recordings (`cw001.wav`–`cw200.wav`), archived as a **held-out real-world eval set** — never used for training. `data/real/labels.csv` holds the 58 known ground-truth transcriptions (extracted from the old `main.org`; the other 142 files have no known label).
- `data/noise/` — real environmental noise beds (`anthro/`, `chords/`, `notes/`, `voice/`, `wildlife/`), repurposed from the old presentation demo audio, used by `synth/noise.py` for SNR-controlled mixing.
- `legacy/archive/` — the original Org-mode/R/LaTeX implementation (`main.org`, `functions.org`, `docs/`), preserved for history. See its own README for what's there.

## Gotchas / non-obvious things

- **`data/real/*.wav` are 32-bit PCM, not 8-bit.** The old R analysis (`tuneR`) mis-reported bit depth; direct header inspection confirms 32-bit. `morse.io.wav.read_wav` derives dtype from the actual file header — don't hardcode assumptions from the archived R output.
- **Only 58 of the 200 real recordings have known labels** (`data/real/labels.csv`), sourced from the Kaggle "morse-challenge" competition via the old `main.org`. The other 142 have no ground truth.
- **Synthetic audio spec is 16kHz mono float32** (`morse.synth.generator.DEFAULT_SAMPLE_RATE`), independent of the real dataset's 8kHz — chosen for the target embedded deployment, not for parity with `data/real/`.
- **The mel-spectrogram module (`features/melspec.py`) and the CTC blank-token placement (`labels/vocab.py`) are deliberately designed to be shared unchanged with future training/inference code** — changing their defaults has downstream consequences beyond this repo's current scope.
- **`requirements.txt` vs `requirements-ml.txt` is a hard boundary**, not just organizational tidiness: everything except `morse.data.pipeline` must work without TensorFlow installed, so that dataset generation and DSP work stay lightweight.

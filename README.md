# NeuralRysum

A real-time neural instrument: train it on a set of recordings, then play that
sound from live audio, from MIDI, or with no input at all, with direct control
over timbre.

- **Split-rate model.** A fast path (64 samples, 1.33 ms) carries what is
  played; a slow path carries timbre and steers the decoder. Latency depends
  only on the fast path.
- **Three sources.** Audio in, MIDI (any tuning, quarter-comma meantone built
  in), or free generation from a prior.
- **Timbre control.** An 8-dimensional latent on an XY pad and knobs, plus
  Brightness / Noisiness / Spread with fixed meaning. OSC out of all of it.
- **Own inference engine.** Header-only C++, no libtorch, checked sample by
  sample against PyTorch.

Design and open questions: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Status.** The code is complete and tested for correctness (causality,
PyTorch/C++ parity, latency). No model has been trained on real recordings
yet, so nothing is known about sound quality. The plugin sources type-check
against JUCE 8.0.4 but have not been built or run in a host yet.

## 1. Install

```
pip install torch            # for a GPU build see pytorch.org
pip install -e .[dev]
pytest                       # needs g++ or clang++ on PATH for the engine tests
```

## 2. Train

Put recordings (wav / flac / aiff, any channel count, ideally 48 kHz) in a
folder. As with RAVE, think in hours of material, not minutes.

```
python -m neuralrysum.train --data D:\organ --out runs\rysum
python -m neuralrysum.train --data D:\organ --out runs\rysum_large --size large
```

Training resumes from `runs\<name>\last.pt` when started again. Losses go to
the console and `log.csv`. The first 100k steps are spectral-only; the
discriminator joins after that (`--warmup_adv`).

## 3. Listen and measure

```
python -m neuralrysum.infer --ckpt runs\rysum\last.pt --input guitar.wav --out out.wav
python -m neuralrysum.infer --ckpt runs\rysum\last.pt --free 20 --out free.wav
python -m neuralrysum.latency --ckpt runs\rysum\last.pt
```

## 4. Export for the plugin

```
python -m neuralrysum.export --ckpt runs\rysum\last.pt --out rysum.nrys --data D:\organ
```

`--data` analyses the dataset so the latent knobs follow its main axes of
variation. Leave it out for raw latent dimensions.

## 5. Build the plugin (Windows)

Needs Visual Studio 2022 (Desktop development with C++), CMake and Git.

```
cd plugin
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

Copy `plugin\build\NeuralRysum_artefacts\Release\VST3\NeuralRysum.vst3` to
`C:\Program Files\Common Files\VST3`, rescan in Reaper, insert it on a track
and press **Load model...**. A standalone app is built next to it.

Run the host at 48 kHz (the model's rate). For zero added latency use a block
size that is a multiple of 64. The build uses AVX2.

JUCE is downloaded during configuration; it is licensed AGPLv3 or commercially.

## Plugin controls

| Control | What it does |
|---|---|
| Source | Audio, MIDI, Audio + MIDI, Free |
| Timbre | Manual (knobs set the timbre) or Follow input (knobs are offsets) |
| XY pad, Latent 3-8 | Position in the learned timbre space, in standard deviations |
| Brightness, Noisiness, Spread | Explicit descriptors, roughly -1 .. 1 |
| Glide | Smoothing of all timbre controls |
| Temperature | Free mode: how much the prior wanders (0 = deterministic) |
| Model noise | Level of the noise the decoder shapes into wind and chiff |
| MIDI breath | Noise mixed into the MIDI exciter |
| Tuning, Meantone root, A4 | Equal or quarter-comma meantone for MIDI |
| OSC out, OSC port | Sends `/neuralrysum/cond` (8 latents + 3 descriptors) to localhost |
| Capture followed timbre | Copies the timbre being followed onto the knobs |

## Layout

```
neuralrysum/   model, training, export, offline tools (PyTorch)
engine/        real-time C++ engine (header-only) and its command line harness
plugin/        JUCE plugin (VST3 + standalone)
tests/         causality, parity, latency, tuning and training tests
docs/          architecture and open questions
```

Engine benchmark: build `engine/` with CMake, then `nrys_cli bench model.nrys`.

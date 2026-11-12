# NeuralRysum: architecture

A neural audio model and plugin for playing a trained sound (for example an
organ) in real time from an instrument, from MIDI, or with no input at all,
with direct control over timbre and a fast path short enough to feel immediate.

## The idea: two rates

RAVE and BRAVE push everything through one latent at one rate. That ties two
things together that do not belong together: a latent that is abstract enough
to be a good timbre control has to be heavily compressed, and compression is
what costs latency (one latent frame per N samples means waiting N samples).

NeuralRysum splits the signal into two paths that run at different rates.

| | Fast path | Slow path |
|---|---|---|
| Carries | what is played: pitch, phase, envelope, onsets | what it sounds like: timbre, registration |
| Rate | every 64 samples (1.33 ms at 48 kHz) | every 2048 samples (42.7 ms) |
| Size | 12 numbers per frame (`e`) | 8 latent dims (`z`) + 3 descriptors |
| May lag? | no, this is the latency | yes, by a slow frame or two, inaudibly |

The slow path does not get concatenated to the fast code. It modulates the
decoder (FiLM: a scale and an offset per channel in every decoder block), so
the timbre controls act on *how* the decoder renders the fast code.

```
 audio in / MIDI exciter ─► FastEncoder ─► e (64-sample hop) ───────────┐
                                │ trunk features                        ▼
                                ▼                                    Decoder ─► audio out
                            SlowEncoder ─► z ─┐                         ▲
 knobs: latent axes, brightness, ... ─────────┴─► Conditioner ─► FiLM ──┘

 Free mode: Prior p(e_t | e_<t, conditioning) replaces the encoder.
```

## Three ways to drive it

**Audio.** The input (guitar, voice, anything) goes through the fast encoder.
Polyphony and any tuning pass through untouched, because pitch is never
extracted or quantised: the fast code is a learned description of the waveform.

**MIDI.** A bank of 16 band-limited saws stands in for the instrument and is
fed to the same fast encoder. No paired MIDI/audio data is needed to train
this, and pitch comes from a 128-entry frequency table, so quarter-comma
meantone (any root, any A4) or any other tuning is exact. `Audio + MIDI` sums
both.

**Free.** A small autoregressive prior (a GRU over the fast code, conditioned
on the timbre controls) invents the fast code frame by frame. Temperature
controls how far it wanders. Moving the latent knobs steers it while it runs.

In every mode the timbre comes from the same place: either the knobs
(**Manual**) or the slow encoder listening to the input (**Follow**, where the
knobs become offsets). "Capture followed timbre" copies what is being followed
onto the knobs.

## Timbre controls

- **Latent 1-8.** The learned timbre space. If the model is exported with
  `--data`, the axes are the principal axes of your dataset, ordered by how
  much they vary, and scaled so that 1.0 is one standard deviation. Latent 1
  and 2 are on the XY pad.
- **Brightness, Noisiness, Spread.** Measured on the training audio (spectral
  centroid, flatness, bandwidth) and given to the decoder as explicit inputs,
  so these three knobs have a fixed meaning for every model.
- **Glide.** Smoothing time of all timbre controls.
- **OSC out.** The current conditioning vector is sent as
  `/neuralrysum/cond f f f ...` (8 latent values, then 3 descriptors) to
  `127.0.0.1:<port>`, 30 times a second, for mapping onto spatialisation.

## Latency

Every layer is causal: no look-ahead anywhere, no PQMF, no transposed
convolutions. The only structural delay is filling one 64-sample frame.

| Situation | Added latency |
|---|---|
| Host block is a multiple of 64 (64, 128, 256, ...) | 0 samples. The frame containing an input sample already reacts to it. |
| Any other block size | exactly 64 samples (1.33 ms), reported to the host |
| MIDI | note-ons are moved to the start of their frame (up to 1.33 ms early) |

On top of that comes your audio interface's buffer, and whatever delay the
network *learns*. A causal network can still choose to respond late.
`python -m neuralrysum.latency --ckpt ...` measures that on a trained model.

The slow path is one slow frame late by construction (a slow frame is only
known once its 2048 samples have arrived). Training applies the same delay,
so the plugin and the training graph are identical.

## Training

One stage, one script.

- **Reconstruction**: multi-resolution STFT loss, including 128- and
  256-sample windows that punish late or smeared onsets.
- **Timbre perturbation**: the fast encoder mostly hears a copy of the target
  with a random EQ, random saturation and a little noise, while the decoder
  must still produce the clean target and the slow encoder hears the clean
  target. The fast code therefore cannot rely on timbre. This is what should
  let a guitar or the saw exciter drive an organ model.
- **KL** on `z` (small weight, warmed up) keeps the latent smooth and navigable.
- **Descriptor head**: the slow encoder also predicts the three descriptors,
  so Follow mode needs no FFT in the plugin.
- **Prior**: trained in the same run on detached fast codes.
- **Adversarial phase** (after `warmup_adv` steps): multi-resolution
  spectrogram discriminators with hinge loss and feature matching.

## Real-time engine

`engine/include/nrys/` is a header-only C++17 implementation of the whole
model with no dependencies: no libtorch, no ONNX Runtime. It allocates
everything at load time and neither allocates nor locks while processing.
The `.nrys` file holds the configuration and the raw weights.

The engine is tested sample-by-sample against PyTorch (`tests/test_engine.py`)
for Manual, Follow and Free, with block sizes 1, 37, 64, 256, 480 and 512.

Default model: about 1.0 M parameters. `--size large`: about 3.2 M, same
latency, roughly three times the CPU.

## What is established and what is not

Verified by the test suite:

- the network has zero look-ahead, and reacts within the frame of a change;
- the slow path never uses the current slow frame;
- the C++ engine equals the PyTorch model to within 2e-4;
- added latency is 0 or exactly 64 samples as described above;
- the meantone table has pure thirds and quarter-comma fifths;
- the training loop runs end to end, including the adversarial phase and resume.
- the training loop learns: on a synthetic three-note chord the output
  reaches the target's harmonics to within about 3 dB after 900 CPU steps
  (a plumbing check, not a statement about quality).

CPU, measured on one core of a shared 2.1 GHz Xeon with AVX2: about 20 % for
the default model and about 56 % for `large`. Single frames occasionally
overran the 1.33 ms budget on that shared machine, so glitch-free playback at
a 64-sample buffer still has to be confirmed on real hardware. Larger host
buffers (128, 256) hide such spikes without adding model latency.

Not established yet. These are the research questions of this project:

1. **Sound quality.** No model has been trained on real recordings. The
   default model is much smaller than RAVE; it may need `--size large`.
2. **Does timbre perturbation separate the two paths well enough?** If the
   fast code still leaks timbre, guitar and MIDI input will sound like a
   filtered guitar / saw rather than the trained sound. The lever is the
   perturbation strength (`augment.py`) and `fast_dim`.
3. **MIDI through a saw exciter** is the least certain mode. If it is weak,
   the next step is to add synthetic exciter signals to the perturbation.
4. **Free mode** will give evolving textures and drones. A 128-unit GRU has
   no long-term musical structure.
5. **Descriptor knobs** are not adversarially removed from `z`, so `z` may
   duplicate some brightness. If the knobs feel weak, add that adversary.
6. **Learned latency** has to be measured after training.

## Roadmap

1. Train on the organ recordings; listen; measure latency.
2. Tune the split (points 2 and 3 above).
3. Multichannel output (one decoder head per Ambisonics channel or per pipe division).
4. Scala file import for arbitrary tunings; MPE.
5. Sample-rate conversion in the plugin for hosts not running at the model rate.

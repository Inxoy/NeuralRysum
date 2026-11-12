// NeuralRysum real-time engine.
//
// Mirrors neuralrysum/model.py frame by frame (one frame = `hop` samples,
// 64 by default). Usage:
//
//     nrys::Engine eng;
//     std::string err;
//     eng.load(bytes, size, err);          // message thread
//     eng.prepare(sampleRate, aligned);    // before playback
//     eng.process(in, out, numSamples);    // audio thread: no alloc, no locks
//
// Latency of the audio path:
//   aligned  (every block is a multiple of hop): 0 samples added. The frame
//            that contains an input sample already reacts to it.
//   buffered (any block size): exactly `hop` samples.
// MIDI and Free have no input to wait for.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "Exciter.h"
#include "Layers.h"
#include "Model.h"

namespace nrys {

enum class Source { Audio = 0, Midi = 1, AudioPlusMidi = 2, Free = 3 };
enum class TimbreMode { Manual = 0, Follow = 1 };

class Engine {
public:
    // ---- setup (not real-time safe) ---------------------------------------
    bool load(const void* bytes, size_t size, std::string& err) {
        ModelFile m;
        if (!m.loadFromMemory(bytes, size, err)) return false;
        try {
            build(m);
        } catch (const std::exception& ex) {
            err = ex.what();
            loaded = false;
            return false;
        }
        loaded = true;
        prepare(static_cast<double>(modelRate), true);
        return true;
    }

    bool loadFile(const std::string& path, std::string& err) {
        std::ifstream f(path, std::ios::binary);
        if (!f) { err = "cannot open " + path; return false; }
        std::vector<char> bytes((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        return load(bytes.data(), bytes.size(), err);
    }

    // aligned = the host promises blocks that are multiples of hop().
    void prepare(double sampleRate, bool alignedBlocks) {
        hostRate = sampleRate;
        aligned = alignedBlocks;
        exciter.prepare(sampleRate);
        setSmoothingMs(smoothingMs);
        reset();
    }

    void reset() {
        if (!loaded) return;
        for (auto& c : encConv) c.reset();
        for (auto& c : encRes) c.reset();
        toE.reset();
        for (auto& c : slowConv) c.reset();
        std::fill(slowCount.begin(), slowCount.end(), 0);
        slowGru.reset();
        inConv.reset();
        for (auto& c : upConv) c.reset();
        for (auto& c : blockConv) c.reset();
        outConv.reset();
        priorGru.reset();
        std::fill(ePrev.begin(), ePrev.end(), 0.0f);
        std::fill(followed.begin(), followed.end(), 0.0f);
        std::fill(inBuf.begin(), inBuf.end(), 0.0f);
        std::fill(outBuf.begin(), outBuf.end(), 0.0f);
        fill = 0;
        condValid = false;
        condPrimed = false;
    }

    // ---- info --------------------------------------------------------------
    bool isLoaded() const noexcept { return loaded; }
    int hop() const noexcept { return hopSize; }
    int sampleRate() const noexcept { return modelRate; }
    int latentDim() const noexcept { return zDim; }
    int numDescriptors() const noexcept { return nDesc; }
    int latencySamples() const noexcept { return aligned ? 0 : hopSize; }
    bool isAligned() const noexcept { return aligned; }

    // ---- controls (call from the audio thread, between process calls) ------
    void setSource(Source s) noexcept { source = s; }
    void setTimbreMode(TimbreMode m) noexcept {
        if (m != timbreMode) {
            timbreMode = m;
            for (auto& c : slowConv) c.reset();
            std::fill(slowCount.begin(), slowCount.end(), 0);
            slowGru.reset();
            std::fill(followed.begin(), followed.end(), 0.0f);
        }
    }
    // Latent knob i, in standard deviations along principal axis i of the
    // training data. In Follow mode the knobs are offsets to the followed timbre.
    void setLatent(int i, float v) noexcept { if (i >= 0 && i < zDim) knobs[i] = v; }
    // Descriptor knob: 0 brightness, 1 noisiness, 2 spread; roughly [-1, 1].
    void setDescriptor(int i, float v) noexcept { if (i >= 0 && i < nDesc) descKnobs[i] = v; }
    void setTemperature(float t) noexcept { temperature = t; }  // Free mode
    void setNoiseAmount(float a) noexcept { noiseAmount = a; }   // 1 = as trained
    void setSmoothingMs(float ms) noexcept {
        smoothingMs = ms;
        const float frames = ms * 0.001f * static_cast<float>(hostRate) / static_cast<float>(hopSize);
        smoothCoef = frames > 1e-3f ? 1.0f - std::exp(-1.0f / frames) : 1.0f;
    }

    Exciter& midi() noexcept { return exciter; }

    // Current conditioning vector [z..., descriptors...] after smoothing, for
    // meters / OSC output. Safe to read from another thread (plain floats).
    const std::vector<float>& conditioning() const noexcept { return cond; }

    // Copy the timbre currently being followed into the manual knobs' space.
    // Returns latent knob values in `outKnobs` (size latentDim()).
    void captureFollowed(float* outKnobs, float* outDesc) const noexcept {
        // knobs = B^-T (z - mean); B has orthogonal rows (scaled PCs), so
        // project on each row and divide by its squared length.
        for (int i = 0; i < zDim; ++i) {
            float dot = 0.0f, len = 0.0f;
            for (int j = 0; j < zDim; ++j) {
                const float bij = zBasis[static_cast<size_t>(i) * zDim + j];
                dot += bij * (followed[j] - zMean[j]);
                len += bij * bij;
            }
            outKnobs[i] = len > 1e-12f ? dot / len : 0.0f;
        }
        for (int i = 0; i < nDesc; ++i) outDesc[i] = followed[zDim + i];
    }

    // ---- audio -------------------------------------------------------------
    // in may be nullptr (treated as silence). in and out may alias.
    void process(const float* in, float* out, int n) noexcept {
        if (!loaded) {
            std::fill(out, out + n, 0.0f);
            return;
        }
        if (aligned && (n % hopSize != 0 || fill != 0)) {
            // the host broke its promise: fall back to the buffered path for good
            aligned = false;
            std::fill(outBuf.begin(), outBuf.end(), 0.0f);
            fill = 0;
        }
        const bool useAudio = source == Source::Audio || source == Source::AudioPlusMidi;
        const bool useMidi = source == Source::Midi || source == Source::AudioPlusMidi;
        if (aligned) {
            for (int o = 0; o < n; o += hopSize) {
                for (int i = 0; i < hopSize; ++i) {
                    float v = (useAudio && in) ? in[o + i] : 0.0f;
                    if (useMidi) v += exciter.tick();
                    inBuf[i] = v;
                }
                runFrame();
                std::copy(outBuf.begin(), outBuf.end(), out + o);
            }
            return;
        }
        for (int i = 0; i < n; ++i) {
            float v = (useAudio && in) ? in[i] : 0.0f;
            if (useMidi) v += exciter.tick();
            out[i] = outBuf[fill];
            inBuf[fill] = v;
            if (++fill == hopSize) {
                runFrame();
                fill = 0;
            }
        }
    }

    // ---- test hooks ----------------------------------------------------------
    // Bypass knobs and smoothing: use exactly this conditioning vector.
    void setRawConditioning(const float* c) noexcept {
        rawCond.assign(c, c + condDim);
        useRawCond = true;
    }
    const std::vector<float>& lastFastCode() const noexcept { return e; }

private:
    // ======================================================================
    void build(const ModelFile& m) {
        modelRate = m.num("sample_rate");
        hopSize = m.num("hop");
        fastDim = m.num("fast_dim");
        zDim = m.num("z_dim");
        nDesc = m.num("n_desc");
        condDim = zDim + nDesc;
        const auto encStrides = m.list("enc_strides");
        const auto slowStrides = m.list("slow_strides");
        decStrides = m.list("dec_strides");
        decCh = m.list("dec_channels");
        const auto dil = m.list("dec_dilations");
        nDil = static_cast<int>(dil.size());

        size_t scratch = static_cast<size_t>(hopSize);
        auto grow = [&](size_t v) { scratch = std::max(scratch, v); };

        // fast encoder
        const int nEnc = static_cast<int>(encStrides.size());
        encConv.assign(nEnc, {});
        encRes.assign(2 * nEnc, {});
        encStride = encStrides;
        int T = hopSize;
        for (int i = 0; i < nEnc; ++i) {
            const std::string p = "fast.convs." + std::to_string(i) + ".conv";
            encConv[i].init(m.t(p + ".weight"), m.t(p + ".bias"), encStrides[i], 1, T);
            T /= encStrides[i];
            const std::string r = "fast.res." + std::to_string(i);
            encRes[2 * i].init(m.t(r + ".conv1.conv.weight"), m.t(r + ".conv1.conv.bias"), 1, 1, T);
            encRes[2 * i + 1].init(m.t(r + ".conv2.conv.weight"), m.t(r + ".conv2.conv.bias"), 1, 1, T);
            grow(static_cast<size_t>(T) * encConv[i].cout);
        }
        if (T != 1) throw std::runtime_error("encoder strides do not multiply to hop");
        trunkCh = encConv.back().cout;
        toE.init(m.t("fast.to_e.conv.weight"), m.t("fast.to_e.conv.bias"), 1, 1, 1);
        e.assign(fastDim, 0.0f);
        ePrev.assign(fastDim, 0.0f);
        ah.assign(trunkCh, 0.0f);

        // slow encoder
        const int nSlow = static_cast<int>(slowStrides.size());
        slowConv.assign(nSlow, {});
        slowStage.assign(nSlow, {});
        slowTmp.assign(nSlow, {});
        slowCount.assign(nSlow, 0);
        slowStride = slowStrides;
        for (int i = 0; i < nSlow; ++i) {
            const std::string p = "slow.convs." + std::to_string(i) + ".conv";
            slowConv[i].init(m.t(p + ".weight"), m.t(p + ".bias"), slowStrides[i], 1, slowStrides[i]);
            slowStage[i].assign(static_cast<size_t>(slowStrides[i]) * slowConv[i].cin, 0.0f);
            slowTmp[i].assign(slowConv[i].cout, 0.0f);
        }
        slowGru.init(m, "slow.gru");
        slowHead.init(m.t("slow.head.weight"), m.t("slow.head.bias"));
        slowOut.assign(slowHead.out, 0.0f);
        followed.assign(condDim, 0.0f);

        // conditioning
        condL1.init(m.t("decoder.cond.l1.weight"), m.t("decoder.cond.l1.bias"));
        condL2.init(m.t("decoder.cond.l2.weight"), m.t("decoder.cond.l2.bias"));
        condHead.init(m.t("decoder.cond.head.weight"), m.t("decoder.cond.head.bias"));
        h1.assign(condL1.out, 0.0f);
        h2.assign(condL2.out, 0.0f);
        film.assign(condHead.out, 0.0f);
        cond.assign(condDim, 0.0f);
        condUsed.assign(condDim, 0.0f);
        target.assign(condDim, 0.0f);
        knobs.assign(zDim, 0.0f);
        descKnobs.assign(nDesc, 0.0f);
        zMean = m.t("z_mean").data;
        zBasis = m.t("z_basis").data;

        // decoder
        inConv.init(m.t("decoder.in_conv.conv.weight"), m.t("decoder.in_conv.conv.bias"), 1, 1, 1);
        const int nStage = static_cast<int>(decCh.size());
        upConv.assign(nStage - 1, {});
        blockConv.assign(static_cast<size_t>(2) * nStage * nDil, {});
        T = 1;
        int bi = 0;
        for (int s = 0; s < nStage; ++s) {
            if (s > 0) {
                const std::string p = "decoder.ups." + std::to_string(s - 1) + ".conv.conv";
                upConv[s - 1].init(m.t(p + ".weight"), m.t(p + ".bias"), 1, 1, T);
                grow(static_cast<size_t>(T) * upConv[s - 1].cout);
                T *= decStrides[s - 1];
            }
            for (int d = 0; d < nDil; ++d, ++bi) {
                const std::string p = "decoder.blocks." + std::to_string(bi);
                blockConv[2 * bi].init(m.t(p + ".conv1.conv.weight"), m.t(p + ".conv1.conv.bias"), 1, dil[d], T);
                blockConv[2 * bi + 1].init(m.t(p + ".conv2.conv.weight"), m.t(p + ".conv2.conv.bias"), 1, 1, T);
            }
            grow(static_cast<size_t>(T) * decCh[s]);
        }
        if (T != hopSize) throw std::runtime_error("decoder strides do not multiply to hop");
        noiseW = m.t("decoder.noise_w").data;
        outConv.init(m.t("decoder.out_conv.conv.weight"), m.t("decoder.out_conv.conv.bias"), 1, 1, hopSize);

        // prior
        priorGru.init(m, "prior.gru");
        priorHead.init(m.t("prior.head.weight"), m.t("prior.head.bias"));
        priorIn.assign(fastDim + condDim, 0.0f);
        priorOut.assign(2 * fastDim, 0.0f);

        for (auto* v : {&sA, &sB, &sC}) v->assign(scratch, 0.0f);
        inBuf.assign(hopSize, 0.0f);
        outBuf.assign(hopSize, 0.0f);
        fill = 0;
    }

    // ======================================================================
    void runFrame() noexcept {
        // conditioning first: a slow frame finished by encode() below is only
        // heard from the next frame on (the one-slow-frame delay of training)
        updateConditioning();
        if (source != Source::Free) encode();
        else priorStep();
        decode();
    }

    void encode() noexcept {
        float* x = sA.data();
        float* y = sB.data();
        float* tmp = sC.data();
        std::copy(inBuf.begin(), inBuf.end(), x);
        int T = hopSize;
        for (size_t i = 0; i < encConv.size(); ++i) {
            if (i > 0) leaky(x, x, T * encConv[i].cin);
            encConv[i].process(x, T, y);
            T /= encStride[i];
            const int n = T * encConv[i].cout;
            leaky(y, tmp, n);
            encRes[2 * i].process(tmp, T, x);
            leaky(x, x, n);
            encRes[2 * i + 1].process(x, T, tmp);
            for (int j = 0; j < n; ++j) y[j] += tmp[j];
            std::swap(x, y);
        }
        leaky(x, ah.data(), trunkCh);
        toE.process(ah.data(), 1, e.data());
        for (auto& v : e) v = std::tanh(v);
        if (timbreMode == TimbreMode::Follow) pushSlow(ah.data());
    }

    // Feed one fast frame of trunk features into the slow path. Each level
    // collects `stride` frames, then emits one frame to the level above.
    void pushSlow(const float* frame) noexcept {
        const float* in = frame;
        for (size_t level = 0; level < slowConv.size(); ++level) {
            Conv1d& c = slowConv[level];
            std::copy(in, in + c.cin,
                      slowStage[level].begin() + static_cast<size_t>(slowCount[level]) * c.cin);
            if (++slowCount[level] < slowStride[level]) return;
            slowCount[level] = 0;
            float* out = slowTmp[level].data();
            c.process(slowStage[level].data(), slowStride[level], out);
            leaky(out, out, c.cout);
            in = out;
        }
        slowGru.step(in);
        slowHead.apply(slowGru.h.data(), slowOut.data());
        // head = [mean (z), logvar (z), descriptors]. This is picked up by the
        // next frame, i.e. one slow frame late, exactly as in training.
        for (int i = 0; i < zDim; ++i) followed[i] = slowOut[i];
        for (int i = 0; i < nDesc; ++i) followed[zDim + i] = slowOut[2 * zDim + i];
    }

    void priorStep() noexcept {
        std::copy(ePrev.begin(), ePrev.end(), priorIn.begin());
        std::copy(cond.begin(), cond.end(), priorIn.begin() + fastDim);
        priorGru.step(priorIn.data());
        priorHead.apply(priorGru.h.data(), priorOut.data());
        for (int i = 0; i < fastDim; ++i) {
            const float logsig = std::min(2.0f, std::max(-7.0f, priorOut[fastDim + i]));
            float v = priorOut[i];
            if (temperature > 0.0f) v += temperature * std::exp(logsig) * gauss();
            e[i] = ePrev[i] = std::min(1.0f, std::max(-1.0f, v));
        }
    }

    void updateConditioning() noexcept {
        if (useRawCond) {
            cond = rawCond;  // same size: no allocation
        } else {
            const bool follow = timbreMode == TimbreMode::Follow && source != Source::Free;
            for (int j = 0; j < zDim; ++j) target[j] = follow ? followed[j] : zMean[j];
            for (int i = 0; i < zDim; ++i) {
                const float k = knobs[i];
                if (k == 0.0f) continue;
                const float* row = zBasis.data() + static_cast<size_t>(i) * zDim;
                for (int j = 0; j < zDim; ++j) target[j] += k * row[j];
            }
            for (int i = 0; i < nDesc; ++i)
                target[zDim + i] = descKnobs[i] + (follow ? followed[zDim + i] : 0.0f);
            const float a = condPrimed ? smoothCoef : 1.0f;
            for (int j = 0; j < condDim; ++j) cond[j] += a * (target[j] - cond[j]);
            condPrimed = true;
        }
        // the FiLM network only has to run when the conditioning moved
        float diff = 0.0f;
        for (int j = 0; j < condDim; ++j) diff = std::max(diff, std::fabs(cond[j] - condUsed[j]));
        if (condValid && diff < 1e-6f) return;
        condUsed = cond;
        condValid = true;
        condL1.apply(cond.data(), h1.data());
        leaky(h1.data(), h1.data(), condL1.out);
        condL2.apply(h1.data(), h2.data());
        leaky(h2.data(), h2.data(), condL2.out);
        condHead.apply(h2.data(), film.data());
    }

    void decode() noexcept {
        float* x = sA.data();
        float* y = sB.data();
        float* tmp = sC.data();
        inConv.process(e.data(), 1, x);
        int T = 1, bi = 0;
        size_t fo = 0;
        const int nStage = static_cast<int>(decCh.size());
        for (int s = 0; s < nStage; ++s) {
            const int ch = decCh[s];
            if (s > 0) {
                const int r = decStrides[s - 1];
                leaky(x, x, T * decCh[s - 1]);
                upConv[s - 1].process(x, T, y);
                // sub-pixel shuffle: conv channel c * r + j at frame t -> channel c at t * r + j
                for (int t = 0; t < T; ++t)
                    for (int c = 0; c < ch; ++c)
                        for (int j = 0; j < r; ++j)
                            x[(static_cast<size_t>(t) * r + j) * ch + c] =
                                y[static_cast<size_t>(t) * ch * r + static_cast<size_t>(c) * r + j];
                T *= r;
            }
            if (s == nStage - 1 && noiseAmount > 0.0f) {
                for (int t = 0; t < T; ++t) {
                    const float nz = noiseAmount * gauss();
                    float* xt = x + static_cast<size_t>(t) * ch;
                    for (int c = 0; c < ch; ++c) xt[c] += noiseW[c] * nz;
                }
            }
            for (int d = 0; d < nDil; ++d, ++bi) {
                const float* g = film.data() + fo;
                const float* b = g + ch;
                fo += static_cast<size_t>(2) * ch;
                const int n = T * ch;
                leaky(x, y, n);
                blockConv[2 * bi].process(y, T, tmp);
                for (int t = 0; t < T; ++t) {
                    float* ht = tmp + static_cast<size_t>(t) * ch;
                    for (int c = 0; c < ch; ++c) {
                        const float v = (1.0f + g[c]) * ht[c] + b[c];
                        ht[c] = v > 0.0f ? v : kLeak * v;
                    }
                }
                blockConv[2 * bi + 1].process(tmp, T, y);
                for (int j = 0; j < n; ++j) x[j] += y[j];
            }
        }
        leaky(x, x, T * decCh.back());
        outConv.process(x, T, y);
        for (int t = 0; t < T; ++t) outBuf[t] = std::tanh(y[t]);
    }

    float gauss() noexcept {
        // sum of uniforms is plenty for excitation noise and has no log/sqrt
        float s = 0.0f;
        for (int i = 0; i < 4; ++i) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            s += static_cast<float>(rng) * (1.0f / 4294967296.0f);
        }
        return (s - 2.0f) * 1.7320508f;  // variance of 4 uniforms = 1/3
    }

    // ---- state -------------------------------------------------------------
    bool loaded = false, aligned = true;
    int modelRate = 48000, hopSize = 64, fastDim = 0, zDim = 0, nDesc = 0, condDim = 0;
    int trunkCh = 0, nDil = 0;
    double hostRate = 48000.0;

    std::vector<Conv1d> encConv, encRes, slowConv, upConv, blockConv;
    Conv1d toE, inConv, outConv;
    std::vector<int> encStride, slowStride, decStrides, decCh, slowCount;
    std::vector<std::vector<float>> slowStage, slowTmp;
    GRU slowGru, priorGru;
    Linear slowHead, condL1, condL2, condHead, priorHead;

    std::vector<float> e, ePrev, ah, slowOut, followed;
    std::vector<float> h1, h2, film, cond, condUsed, target, rawCond;
    std::vector<float> knobs, descKnobs, zMean, zBasis, noiseW;
    std::vector<float> priorIn, priorOut;
    std::vector<float> sA, sB, sC, inBuf, outBuf;
    int fill = 0;

    Source source = Source::Audio;
    TimbreMode timbreMode = TimbreMode::Manual;
    float temperature = 1.0f, noiseAmount = 1.0f, smoothingMs = 30.0f, smoothCoef = 1.0f;
    bool condValid = false, condPrimed = false, useRawCond = false;
    Exciter exciter;
    uint32_t rng = 0x2545F491u;
};

}  // namespace nrys

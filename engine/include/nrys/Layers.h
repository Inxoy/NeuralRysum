// NeuralRysum real-time engine - streaming layers.
// Mirrors neuralrysum/layers.py. All buffers are allocated in init();
// process() never allocates and never locks.
#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#include "Model.h"

namespace nrys {

constexpr float kLeak = 0.2f;

inline void leaky(const float* src, float* dst, int n) noexcept {
    for (int i = 0; i < n; ++i) dst[i] = src[i] > 0.0f ? src[i] : kLeak * src[i];
}

// Causal conv, frames interleaved as [time][channel].
// Keeps its own history, so consecutive process() calls continue the stream.
struct Conv1d {
    int cin = 0, cout = 0, k = 0, stride = 1, dil = 1, pad = 0;
    std::vector<float> w;     // [k][cin][cout]  (torch stores [cout][cin][k])
    std::vector<float> b;
    std::vector<float> hist;  // last `pad` input frames
    std::vector<float> buf;   // hist + current input

    void init(const Tensor& W, const Tensor& B, int stride_, int dil_, int maxFrames) {
        if (W.shape.size() != 3) throw std::runtime_error("conv weight must be 3-D");
        cout = W.shape[0];
        cin = W.shape[1];
        k = W.shape[2];
        stride = stride_;
        dil = dil_;
        pad = (k - 1) * dil + 1 - stride;
        if (pad < 0) throw std::runtime_error("conv kernel smaller than stride");
        w.resize(static_cast<size_t>(k) * cin * cout);
        for (int co = 0; co < cout; ++co)
            for (int ci = 0; ci < cin; ++ci)
                for (int kk = 0; kk < k; ++kk)
                    w[(static_cast<size_t>(kk) * cin + ci) * cout + co] =
                        W.data[(static_cast<size_t>(co) * cin + ci) * k + kk];
        b = B.data;
        hist.assign(static_cast<size_t>(pad) * cin, 0.0f);
        buf.assign(static_cast<size_t>(pad + maxFrames) * cin, 0.0f);
    }

    void reset() noexcept { std::fill(hist.begin(), hist.end(), 0.0f); }

    // in: T frames of cin (T a multiple of stride). out: T / stride frames of cout.
    void process(const float* in, int T, float* out) noexcept {
        const size_t histBytes = hist.size() * sizeof(float);
        if (pad > 0) std::memcpy(buf.data(), hist.data(), histBytes);
        std::memcpy(buf.data() + hist.size(), in, static_cast<size_t>(T) * cin * sizeof(float));
        const int To = T / stride;
        const float* bias = b.data();
        for (int to = 0; to < To; ++to) {
            float* o = out + static_cast<size_t>(to) * cout;
            for (int co = 0; co < cout; ++co) o[co] = bias[co];
            for (int kk = 0; kk < k; ++kk) {
                const float* x = buf.data() + static_cast<size_t>(to * stride + kk * dil) * cin;
                const float* wk = w.data() + static_cast<size_t>(kk) * cin * cout;
                for (int ci = 0; ci < cin; ++ci) {
                    const float v = x[ci];
                    const float* wc = wk + static_cast<size_t>(ci) * cout;
                    for (int co = 0; co < cout; ++co) o[co] += v * wc[co];
                }
            }
        }
        if (pad > 0)
            std::memcpy(hist.data(), buf.data() + static_cast<size_t>(T) * cin, histBytes);
    }
};

struct Linear {
    int in = 0, out = 0;
    std::vector<float> wt;  // [in][out]
    std::vector<float> b;

    void init(const Tensor& W, const Tensor& B) {
        if (W.shape.size() != 2) throw std::runtime_error("linear weight must be 2-D");
        out = W.shape[0];
        in = W.shape[1];
        wt.resize(static_cast<size_t>(in) * out);
        for (int o = 0; o < out; ++o)
            for (int i = 0; i < in; ++i)
                wt[static_cast<size_t>(i) * out + o] = W.data[static_cast<size_t>(o) * in + i];
        b = B.data;
    }

    void apply(const float* x, float* y) const noexcept {
        for (int o = 0; o < out; ++o) y[o] = b[o];
        for (int i = 0; i < in; ++i) {
            const float v = x[i];
            const float* wc = wt.data() + static_cast<size_t>(i) * out;
            for (int o = 0; o < out; ++o) y[o] += v * wc[o];
        }
    }
};

// One GRU layer, PyTorch gate order (r, z, n).
struct GRU {
    int H = 0;
    Linear ih, hh;
    std::vector<float> h, a, c;

    void init(const ModelFile& m, const std::string& prefix) {
        ih.init(m.t(prefix + ".weight_ih_l0"), m.t(prefix + ".bias_ih_l0"));
        hh.init(m.t(prefix + ".weight_hh_l0"), m.t(prefix + ".bias_hh_l0"));
        H = hh.in;
        h.assign(H, 0.0f);
        a.assign(3 * H, 0.0f);
        c.assign(3 * H, 0.0f);
    }

    void reset() noexcept { std::fill(h.begin(), h.end(), 0.0f); }

    static float sigmoid(float v) noexcept { return 1.0f / (1.0f + std::exp(-v)); }

    void step(const float* x) noexcept {
        ih.apply(x, a.data());
        hh.apply(h.data(), c.data());
        for (int i = 0; i < H; ++i) {
            const float r = sigmoid(a[i] + c[i]);
            const float z = sigmoid(a[H + i] + c[H + i]);
            const float n = std::tanh(a[2 * H + i] + r * c[2 * H + i]);
            h[i] = (1.0f - z) * n + z * h[i];
        }
    }
};

}  // namespace nrys

// Command line harness for the engine: parity checks against PyTorch,
// a CPU benchmark, and a MIDI render. Raw float32 mono files in and out.
//
//   nrys_cli parity <model> <in.f32> <out.f32> manual|follow|free <cond.f32> <block>
//   nrys_cli bench  <model>
//   nrys_cli midi   <model> <out.f32> equal|meantone
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nrys/Engine.h"

static std::vector<float> readF32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::cerr << "cannot open " << path << "\n"; std::exit(2); }
    const size_t n = static_cast<size_t>(f.tellg()) / 4;
    std::vector<float> v(n);
    f.seekg(0);
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
    return v;
}

static void writeF32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * 4));
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: see top of nrys_cli.cpp\n"; return 2; }
    const std::string cmd = argv[1];
    nrys::Engine eng;
    std::string err;
    if (!eng.loadFile(argv[2], err)) { std::cerr << "load failed: " << err << "\n"; return 1; }
    const int hop = eng.hop();

    if (cmd == "parity" && argc >= 8) {
        const std::vector<float> in = readF32(argv[3]);
        const std::string mode = argv[5];
        const std::vector<float> cond = readF32(argv[6]);
        const int block = std::atoi(argv[7]);
        eng.prepare(eng.sampleRate(), block % hop == 0);
        eng.setNoiseAmount(0.0f);
        eng.setSmoothingMs(0.0f);
        eng.setTemperature(0.0f);
        if (mode == "follow") eng.setTimbreMode(nrys::TimbreMode::Follow);
        else eng.setRawConditioning(cond.data());
        eng.setSource(mode == "free" ? nrys::Source::Free : nrys::Source::Audio);
        std::vector<float> out(in.size());
        // deliberately awkward chunking: the result must not depend on it
        for (size_t o = 0; o < in.size(); o += static_cast<size_t>(block)) {
            const int n = static_cast<int>(std::min(in.size() - o, static_cast<size_t>(block)));
            eng.process(in.data() + o, out.data() + o, n);
        }
        writeF32(argv[4], out);
        std::cout << "latency " << eng.latencySamples() << "\n";
        return 0;
    }

    if (cmd == "bench") {
        const int sr = eng.sampleRate();
        const int frames = 10 * sr / hop;  // 10 s of audio
        std::vector<float> in(hop), out(hop);
        const char* names[] = {"audio/manual", "audio/follow", "midi 8 voices", "free"};
        for (int mode = 0; mode < 4; ++mode) {
            eng.prepare(sr, true);
            eng.setSource(mode == 2 ? nrys::Source::Midi : mode == 3 ? nrys::Source::Free : nrys::Source::Audio);
            eng.setTimbreMode(mode == 1 ? nrys::TimbreMode::Follow : nrys::TimbreMode::Manual);
            if (mode == 2) for (int n = 0; n < 8; ++n) eng.midi().noteOn(48 + 3 * n, 0.8f);
            uint32_t r = 1;
            double total = 0.0;
            std::vector<double> times;
            times.reserve(static_cast<size_t>(frames));
            for (int f = 0; f < frames; ++f) {
                for (auto& v : in) { r = r * 1664525u + 1013904223u; v = (static_cast<float>(r >> 8) / 8388608.0f - 1.0f) * 0.1f; }
                const auto t0 = std::chrono::steady_clock::now();
                eng.process(in.data(), out.data(), hop);
                const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
                total += dt;
                times.push_back(dt);
            }
            const double frameTime = static_cast<double>(hop) / sr;
            std::sort(times.begin(), times.end());
            const double p99 = times[times.size() * 99 / 100], worst = times.back();
            std::printf("%-14s  mean %5.1f %%   99th percentile %5.1f %%   worst %6.1f %%  (of one core, frame = %.2f ms)\n",
                        names[mode], 100.0 * total / (frames * frameTime), 100.0 * p99 / frameTime,
                        100.0 * worst / frameTime, frameTime * 1e3);
        }
        return 0;
    }

    if (cmd == "midi" && argc >= 5) {
        const int sr = eng.sampleRate();
        eng.prepare(sr, true);
        eng.setSource(nrys::Source::Midi);
        const std::string tuning = argv[4];
        eng.midi().setTuning(tuning == "meantone" ? nrys::Tuning::meantone(440.0f, 0) : nrys::Tuning::equal(440.0f));
        std::vector<float> out(static_cast<size_t>(2 * sr / hop) * hop);
        for (size_t o = 0; o < out.size(); o += static_cast<size_t>(hop)) {
            if (o == 0) for (int n : {60, 64, 67}) eng.midi().noteOn(n, 0.8f);
            if (o == static_cast<size_t>(sr / hop) * hop) eng.midi().allNotesOff();
            eng.process(nullptr, out.data() + o, hop);
        }
        writeF32(argv[3], out);
        return 0;
    }

    if (cmd == "tuning") {
        const auto eq = nrys::Tuning::equal(440.0f), mt = nrys::Tuning::meantone(440.0f, 0);
        for (int n = 60; n <= 72; ++n) std::printf("%d %.4f %.4f\n", n, eq.hz[n], mt.hz[n]);
        return 0;
    }

    std::cerr << "unknown command\n";
    return 2;
}

// NeuralRysum real-time engine - MIDI exciter and tunings.
//
// In MIDI mode nothing is "synthesised" in the usual sense: a plain bank of
// band-limited saws stands in for the instrument you would otherwise play
// into the plugin, and the model turns it into the trained sound. Pitch comes
// from a table of 128 frequencies in Hz, so any tuning is possible and
// nothing is ever snapped to equal temperament.
#pragma once

#include <array>
#include <cmath>
#include <cstdint>

namespace nrys {

struct Tuning {
    std::array<float, 128> hz{};

    static Tuning equal(float a4 = 440.0f) {
        Tuning t;
        for (int n = 0; n < 128; ++n) t.hz[n] = a4 * std::pow(2.0f, static_cast<float>(n - 69) / 12.0f);
        return t;
    }

    // Quarter-comma meantone: eleven fifths narrowed to 5^(1/4), pure major
    // thirds. `rootPc` (0 = C ... 11 = B) is the centre of the chain of
    // fifths, which runs from three fifths below the root to eight above
    // (Eb ... G# for root C, the usual historical layout; the wolf is G#-Eb).
    // A4 is pinned to `a4`.
    static Tuning meantone(float a4 = 440.0f, int rootPc = 0) {
        static const int fifths[12] = {0, 7, 2, -3, 4, -1, 6, 1, 8, 3, -2, 5};
        const double fifth = 1200.0 * std::log2(std::pow(5.0, 0.25));
        auto cents = [&](int note) {
            const int rel = note - rootPc;
            const int pc = ((rel % 12) + 12) % 12;
            const int oct = (rel - pc) / 12;
            double c = std::fmod(fifths[pc] * fifth, 1200.0);
            if (c < 0) c += 1200.0;
            return oct * 1200.0 + c;
        };
        Tuning t;
        const double ref = cents(69);
        for (int n = 0; n < 128; ++n)
            t.hz[n] = static_cast<float>(a4 * std::pow(2.0, (cents(n) - ref) / 1200.0));
        return t;
    }
};

class Exciter {
public:
    static constexpr int kVoices = 16;

    void prepare(double sampleRate) noexcept {
        sr = static_cast<float>(sampleRate);
        attackStep = 1.0f / (0.003f * sr);
        releaseCoef = std::exp(-1.0f / (0.040f * sr));
        allNotesOff();
        for (auto& v : voices) v = Voice{};
    }

    void setTuning(const Tuning& t) noexcept {
        tuning = t;
        for (auto& v : voices)
            if (v.note >= 0) v.baseInc = tuning.hz[v.note] / sr;
    }

    void setPitchBend(float semitones) noexcept { bend = std::pow(2.0f, semitones / 12.0f); }
    void setBreath(float amount) noexcept { breath = amount; }  // noise mixed into each voice
    void setLevel(float l) noexcept { level = l; }

    void noteOn(int note, float velocity) noexcept {
        if (note < 0 || note > 127) return;
        Voice* pick = nullptr;
        for (auto& v : voices)
            if (v.env <= 0.0f && !v.gate) { pick = &v; break; }
        if (!pick) {  // steal the oldest
            pick = &voices[0];
            for (auto& v : voices)
                if (v.age < pick->age) pick = &v;
        }
        pick->note = note;
        pick->baseInc = tuning.hz[note] / sr;
        pick->amp = velocity;
        pick->gate = true;
        pick->age = ++clock;
        // the envelope and phase carry on if the voice was stolen: no click
    }

    void noteOff(int note) noexcept {
        for (auto& v : voices)
            if (v.note == note) v.gate = false;
    }

    void allNotesOff() noexcept {
        for (auto& v : voices) v.gate = false;
    }

    bool active() const noexcept {
        for (auto& v : voices)
            if (v.gate || v.env > 0.0f) return true;
        return false;
    }

    float tick() noexcept {
        float out = 0.0f;
        for (auto& v : voices) {
            if (!v.gate && v.env <= 0.0f) continue;
            if (v.gate) {
                v.env = v.env + attackStep < 1.0f ? v.env + attackStep : 1.0f;
            } else {
                v.env *= releaseCoef;
                if (v.env < 1e-4f) { v.env = 0.0f; v.note = -1; continue; }
            }
            const float dt = v.baseInc * bend;
            float s = 2.0f * v.phase - 1.0f;
            if (v.phase < dt) {
                const float x = v.phase / dt;
                s -= x + x - x * x - 1.0f;
            } else if (v.phase > 1.0f - dt) {
                const float x = (v.phase - 1.0f) / dt;
                s -= x * x + x + x + 1.0f;
            }
            v.phase += dt;
            if (v.phase >= 1.0f) v.phase -= 1.0f;
            out += v.env * v.amp * (s + breath * white());
        }
        return out * level;
    }

private:
    struct Voice {
        int note = -1;
        float phase = 0.0f, baseInc = 0.0f, amp = 0.0f, env = 0.0f;
        bool gate = false;
        uint64_t age = 0;
    };

    float white() noexcept {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        return static_cast<float>(rng) * (2.0f / 4294967296.0f) - 1.0f;
    }

    std::array<Voice, kVoices> voices{};
    Tuning tuning = Tuning::equal();
    float sr = 48000.0f, attackStep = 0.01f, releaseCoef = 0.999f;
    float bend = 1.0f, breath = 0.02f, level = 0.25f;
    uint64_t clock = 0;
    uint32_t rng = 0x9E3779B9u;
};

}  // namespace nrys

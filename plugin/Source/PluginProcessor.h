#pragma once

#include <juce_audio_utils/juce_audio_utils.h>
#include <juce_osc/juce_osc.h>

#include "nrys/Engine.h"

namespace ids {
constexpr int kLatents = 8;
inline juce::String latent(int i) { return "lat" + juce::String(i + 1); }
static const char* const desc[3] = {"bright", "noisy", "spread"};
}  // namespace ids

class NeuralRysumProcessor : public juce::AudioProcessor,
                             private juce::AsyncUpdater,
                             private juce::Timer {
public:
    NeuralRysumProcessor();
    ~NeuralRysumProcessor() override;

    void prepareToPlay(double sampleRate, int samplesPerBlock) override;
    void releaseResources() override {}
    bool isBusesLayoutSupported(const BusesLayout& layouts) const override;
    void processBlock(juce::AudioBuffer<float>&, juce::MidiBuffer&) override;

    juce::AudioProcessorEditor* createEditor() override;
    bool hasEditor() const override { return true; }

    const juce::String getName() const override { return "NeuralRysum"; }
    bool acceptsMidi() const override { return true; }
    bool producesMidi() const override { return false; }
    bool isMidiEffect() const override { return false; }
    double getTailLengthSeconds() const override { return 0.5; }

    int getNumPrograms() override { return 1; }
    int getCurrentProgram() override { return 0; }
    void setCurrentProgram(int) override {}
    const juce::String getProgramName(int) override { return {}; }
    void changeProgramName(int, const juce::String&) override {}

    void getStateInformation(juce::MemoryBlock& destData) override;
    void setStateInformation(const void* data, int sizeInBytes) override;

    // ---- message thread ----------------------------------------------------
    // Returns an empty string on success, otherwise what went wrong.
    juce::String loadModel(const juce::File& file);
    juce::String describeModel() const;
    // Copy the timbre currently followed from the input onto the knobs.
    void captureFollowedTimbre();

    juce::AudioProcessorValueTreeState apvts;

private:
    static juce::AudioProcessorValueTreeState::ParameterLayout createLayout();
    void pushParameters(nrys::Engine& e);
    void handleMidi(nrys::Engine& e, const juce::MidiMessage& m);
    void handleAsyncUpdate() override;
    void timerCallback() override;

    std::unique_ptr<nrys::Engine> engine;  // swapped under engineLock
    juce::SpinLock engineLock;
    juce::File modelFile;

    double hostRate = 48000.0;
    int hostBlock = 512;
    std::atomic<int> wantedLatency{0};

    // parameter handles
    std::atomic<float>* pSource = nullptr;
    std::atomic<float>* pTimbre = nullptr;
    std::atomic<float>* pLatent[ids::kLatents] = {};
    std::atomic<float>* pDesc[3] = {};
    std::atomic<float>* pTemp = nullptr;
    std::atomic<float>* pSmooth = nullptr;
    std::atomic<float>* pNoise = nullptr;
    std::atomic<float>* pBreath = nullptr;
    std::atomic<float>* pTuning = nullptr;
    std::atomic<float>* pA4 = nullptr;
    std::atomic<float>* pRoot = nullptr;
    std::atomic<float>* pMix = nullptr;
    std::atomic<float>* pGain = nullptr;
    std::atomic<float>* pOscOn = nullptr;
    std::atomic<float>* pOscPort = nullptr;

    // audio-thread state
    int lastTuning = -1, lastRoot = -1;
    float lastA4 = 0.0f;
    std::vector<float> mono, wet, dryDelay;
    int dryPos = 0;
    juce::SmoothedValue<float> gainSmoothed, mixSmoothed;

    // OSC out of the conditioning vector (for spatialisation etc.)
    juce::OSCSender osc;
    int oscConnectedPort = 0;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralRysumProcessor)
};

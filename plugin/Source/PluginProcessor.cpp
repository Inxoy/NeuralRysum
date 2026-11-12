#include "PluginProcessor.h"

#include "PluginEditor.h"

using APVTS = juce::AudioProcessorValueTreeState;

APVTS::ParameterLayout NeuralRysumProcessor::createLayout() {
    APVTS::ParameterLayout l;
    auto pid = [](const juce::String& id) { return juce::ParameterID{id, 1}; };
    auto range = [](float lo, float hi, float step = 0.0f) {
        return juce::NormalisableRange<float>(lo, hi, step);
    };

    l.add(std::make_unique<juce::AudioParameterChoice>(
        pid("source"), "Source", juce::StringArray{"Audio", "MIDI", "Audio + MIDI", "Free"}, 0));
    l.add(std::make_unique<juce::AudioParameterChoice>(
        pid("timbre"), "Timbre", juce::StringArray{"Manual", "Follow input"}, 0));
    for (int i = 0; i < ids::kLatents; ++i)
        l.add(std::make_unique<juce::AudioParameterFloat>(
            pid(ids::latent(i)), "Latent " + juce::String(i + 1), range(-3.0f, 3.0f), 0.0f));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("bright"), "Brightness", range(-1.5f, 1.5f), 0.0f));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("noisy"), "Noisiness", range(-1.5f, 1.5f), 0.0f));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("spread"), "Spread", range(-1.5f, 1.5f), 0.0f));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("temp"), "Temperature", range(0.0f, 2.0f), 1.0f));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("smooth"), "Glide (ms)", range(0.0f, 2000.0f, 1.0f), 30.0f));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("noise"), "Model noise", range(0.0f, 2.0f), 1.0f));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("breath"), "MIDI breath", range(0.0f, 0.5f), 0.02f));
    l.add(std::make_unique<juce::AudioParameterChoice>(
        pid("tuning"), "Tuning", juce::StringArray{"Equal", "Quarter-comma meantone"}, 0));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("a4"), "A4 (Hz)", range(380.0f, 480.0f, 0.1f), 440.0f));
    l.add(std::make_unique<juce::AudioParameterChoice>(
        pid("root"), "Meantone root",
        juce::StringArray{"C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"}, 0));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("mix"), "Dry / wet", range(0.0f, 1.0f), 1.0f));
    l.add(std::make_unique<juce::AudioParameterFloat>(pid("gain"), "Output (dB)", range(-24.0f, 12.0f, 0.1f), 0.0f));
    l.add(std::make_unique<juce::AudioParameterBool>(pid("oscOn"), "OSC out", false));
    l.add(std::make_unique<juce::AudioParameterInt>(pid("oscPort"), "OSC port", 1024, 65535, 9000));
    return l;
}

NeuralRysumProcessor::NeuralRysumProcessor()
    : AudioProcessor(BusesProperties()
                         .withInput("Input", juce::AudioChannelSet::stereo(), true)
                         .withOutput("Output", juce::AudioChannelSet::stereo(), true)),
      apvts(*this, nullptr, "state", createLayout()) {
    pSource = apvts.getRawParameterValue("source");
    pTimbre = apvts.getRawParameterValue("timbre");
    for (int i = 0; i < ids::kLatents; ++i) pLatent[i] = apvts.getRawParameterValue(ids::latent(i));
    for (int i = 0; i < 3; ++i) pDesc[i] = apvts.getRawParameterValue(ids::desc[i]);
    pTemp = apvts.getRawParameterValue("temp");
    pSmooth = apvts.getRawParameterValue("smooth");
    pNoise = apvts.getRawParameterValue("noise");
    pBreath = apvts.getRawParameterValue("breath");
    pTuning = apvts.getRawParameterValue("tuning");
    pA4 = apvts.getRawParameterValue("a4");
    pRoot = apvts.getRawParameterValue("root");
    pMix = apvts.getRawParameterValue("mix");
    pGain = apvts.getRawParameterValue("gain");
    pOscOn = apvts.getRawParameterValue("oscOn");
    pOscPort = apvts.getRawParameterValue("oscPort");
    startTimerHz(30);
}

NeuralRysumProcessor::~NeuralRysumProcessor() {
    stopTimer();
    cancelPendingUpdate();
}

bool NeuralRysumProcessor::isBusesLayoutSupported(const BusesLayout& layouts) const {
    const auto out = layouts.getMainOutputChannelSet();
    const auto in = layouts.getMainInputChannelSet();
    const bool outOk = out == juce::AudioChannelSet::mono() || out == juce::AudioChannelSet::stereo();
    const bool inOk = in.isDisabled() || in == juce::AudioChannelSet::mono() ||
                      in == juce::AudioChannelSet::stereo();
    return outOk && inOk;
}

void NeuralRysumProcessor::prepareToPlay(double sampleRate, int samplesPerBlock) {
    hostRate = sampleRate;
    hostBlock = samplesPerBlock;
    mono.assign(static_cast<size_t>(samplesPerBlock), 0.0f);
    wet.assign(static_cast<size_t>(samplesPerBlock), 0.0f);
    gainSmoothed.reset(sampleRate, 0.02);
    mixSmoothed.reset(sampleRate, 0.02);
    lastTuning = -1;

    const juce::SpinLock::ScopedLockType lock(engineLock);
    if (engine != nullptr) {
        // zero added latency if the host block is a whole number of model
        // frames; the engine falls back to one frame of latency if it is not
        engine->prepare(sampleRate, samplesPerBlock % engine->hop() == 0);
        dryDelay.assign(static_cast<size_t>(engine->hop()), 0.0f);
        dryPos = 0;
        wantedLatency = engine->latencySamples();
        setLatencySamples(wantedLatency.load());
    }
}

void NeuralRysumProcessor::pushParameters(nrys::Engine& e) {
    e.setSource(static_cast<nrys::Source>(juce::roundToInt(pSource->load())));
    e.setTimbreMode(static_cast<nrys::TimbreMode>(juce::roundToInt(pTimbre->load())));
    for (int i = 0; i < ids::kLatents; ++i) e.setLatent(i, pLatent[i]->load());
    for (int i = 0; i < 3; ++i) e.setDescriptor(i, pDesc[i]->load());
    e.setTemperature(pTemp->load());
    e.setSmoothingMs(pSmooth->load());
    e.setNoiseAmount(pNoise->load());
    e.midi().setBreath(pBreath->load());

    const int tuning = juce::roundToInt(pTuning->load());
    const int root = juce::roundToInt(pRoot->load());
    const float a4 = pA4->load();
    if (tuning != lastTuning || root != lastRoot || a4 != lastA4) {
        e.midi().setTuning(tuning == 1 ? nrys::Tuning::meantone(a4, root) : nrys::Tuning::equal(a4));
        lastTuning = tuning;
        lastRoot = root;
        lastA4 = a4;
    }
}

void NeuralRysumProcessor::handleMidi(nrys::Engine& e, const juce::MidiMessage& m) {
    if (m.isNoteOn())
        e.midi().noteOn(m.getNoteNumber(), m.getFloatVelocity());
    else if (m.isNoteOff())
        e.midi().noteOff(m.getNoteNumber());
    else if (m.isAllNotesOff() || m.isAllSoundOff())
        e.midi().allNotesOff();
    else if (m.isPitchWheel())
        e.midi().setPitchBend(2.0f * static_cast<float>(m.getPitchWheelValue() - 8192) / 8192.0f);
}

void NeuralRysumProcessor::processBlock(juce::AudioBuffer<float>& buffer, juce::MidiBuffer& midi) {
    juce::ScopedNoDenormals noDenormals;
    const int n = buffer.getNumSamples();
    const int numIn = getTotalNumInputChannels();
    const int numOut = getTotalNumOutputChannels();

    const juce::SpinLock::ScopedTryLockType lock(engineLock);
    if (!lock.isLocked() || engine == nullptr || n == 0) {
        // no model (or one is being swapped in right now): pass silence
        buffer.clear();
        return;
    }
    nrys::Engine& e = *engine;
    if (static_cast<int>(mono.size()) < n) {  // host exceeded its promised block size
        mono.resize(static_cast<size_t>(n));
        wet.resize(static_cast<size_t>(n));
    }

    // mono sum of the input
    std::fill(mono.begin(), mono.begin() + n, 0.0f);
    if (numIn > 0) {
        const float scale = 1.0f / static_cast<float>(numIn);
        for (int ch = 0; ch < numIn; ++ch)
            juce::FloatVectorOperations::addWithMultiply(mono.data(), buffer.getReadPointer(ch), scale, n);
    }

    pushParameters(e);

    // Render, applying MIDI events on the way. While the engine runs without
    // added latency, events are moved to the start of their 64-sample frame
    // (at most 1.3 ms early) so the block never has to be cut mid-frame.
    int pos = 0;
    const int hop = e.hop();
    for (const auto meta : midi) {
        int at = juce::jlimit(0, n, meta.samplePosition);
        if (e.isAligned()) at = (at / hop) * hop;
        if (at > pos) {
            e.process(mono.data() + pos, wet.data() + pos, at - pos);
            pos = at;
        }
        handleMidi(e, meta.getMessage());
    }
    if (pos < n) e.process(mono.data() + pos, wet.data() + pos, n - pos);
    midi.clear();

    if (e.latencySamples() != wantedLatency.load()) {
        wantedLatency = e.latencySamples();
        triggerAsyncUpdate();  // tell the host from the message thread
    }

    // dry path, delayed by the engine's latency so the mix stays phase-aligned
    const int delay = e.latencySamples();
    if (delay > 0 && static_cast<int>(dryDelay.size()) >= delay) {
        for (int i = 0; i < n; ++i) {
            const float in = mono[static_cast<size_t>(i)];
            mono[static_cast<size_t>(i)] = dryDelay[static_cast<size_t>(dryPos)];
            dryDelay[static_cast<size_t>(dryPos)] = in;
            if (++dryPos >= delay) dryPos = 0;
        }
    }

    gainSmoothed.setTargetValue(juce::Decibels::decibelsToGain(pGain->load()));
    mixSmoothed.setTargetValue(pMix->load());
    for (int i = 0; i < n; ++i) {
        const float mix = mixSmoothed.getNextValue();
        const size_t k = static_cast<size_t>(i);
        wet[k] = gainSmoothed.getNextValue() * (mix * wet[k] + (1.0f - mix) * mono[k]);
    }
    for (int ch = 0; ch < numOut; ++ch) buffer.copyFrom(ch, 0, wet.data(), n);
}

void NeuralRysumProcessor::handleAsyncUpdate() { setLatencySamples(wantedLatency.load()); }

// ---- model loading -----------------------------------------------------------
juce::String NeuralRysumProcessor::loadModel(const juce::File& file) {
    juce::MemoryBlock bytes;
    if (!file.loadFileAsData(bytes)) return "Could not read " + file.getFullPathName();

    auto fresh = std::make_unique<nrys::Engine>();
    std::string err;
    if (!fresh->load(bytes.getData(), bytes.getSize(), err)) return juce::String(err);
    fresh->prepare(hostRate, hostBlock % fresh->hop() == 0);

    {
        const juce::SpinLock::ScopedLockType lock(engineLock);
        std::swap(engine, fresh);  // the old engine is freed below, outside the lock
        dryDelay.assign(static_cast<size_t>(engine->hop()), 0.0f);
        dryPos = 0;
        lastTuning = -1;
        wantedLatency = engine->latencySamples();
    }
    modelFile = file;
    setLatencySamples(wantedLatency.load());
    return {};
}

juce::String NeuralRysumProcessor::describeModel() const {
    if (engine == nullptr) return "No model loaded";
    juce::String s = modelFile.getFileNameWithoutExtension();
    s << "  |  frame " << juce::String(1000.0 * engine->hop() / engine->sampleRate(), 2) << " ms"
      << "  |  added latency " << juce::String(1000.0 * engine->latencySamples() / hostRate, 2) << " ms";
    if (std::abs(hostRate - engine->sampleRate()) > 1.0)
        s << "  |  WARNING: model is " << engine->sampleRate() << " Hz, host is "
          << juce::roundToInt(hostRate) << " Hz (timbre will be shifted)";
    return s;
}

void NeuralRysumProcessor::captureFollowedTimbre() {
    if (engine == nullptr) return;
    std::vector<float> knobs(static_cast<size_t>(engine->latentDim())), d(static_cast<size_t>(engine->numDescriptors()));
    engine->captureFollowed(knobs.data(), d.data());
    auto set = [this](const juce::String& id, float v) {
        if (auto* p = apvts.getParameter(id)) {
            p->beginChangeGesture();
            p->setValueNotifyingHost(p->convertTo0to1(v));
            p->endChangeGesture();
        }
    };
    for (int i = 0; i < juce::jmin(ids::kLatents, engine->latentDim()); ++i)
        set(ids::latent(i), knobs[static_cast<size_t>(i)]);
    for (int i = 0; i < juce::jmin(3, engine->numDescriptors()); ++i)
        set(ids::desc[i], d[static_cast<size_t>(i)]);
    set("timbre", 0.0f);  // back to Manual, now sitting on the captured timbre
}

// ---- OSC out -------------------------------------------------------------------
void NeuralRysumProcessor::timerCallback() {
    const bool on = pOscOn->load() > 0.5f;
    const int port = juce::roundToInt(pOscPort->load());
    if (!on) {
        if (oscConnectedPort != 0) {
            osc.disconnect();
            oscConnectedPort = 0;
        }
        return;
    }
    if (port != oscConnectedPort) {
        osc.disconnect();
        oscConnectedPort = osc.connect("127.0.0.1", port) ? port : 0;
    }
    if (oscConnectedPort == 0 || engine == nullptr) return;
    // a slightly stale copy of the conditioning vector is fine here
    const std::vector<float> c = engine->conditioning();
    juce::OSCMessage msg("/neuralrysum/cond");
    for (float v : c) msg.addFloat32(v);
    osc.send(msg);
}

// ---- state ---------------------------------------------------------------------
void NeuralRysumProcessor::getStateInformation(juce::MemoryBlock& destData) {
    auto state = apvts.copyState();
    state.setProperty("modelPath", modelFile.getFullPathName(), nullptr);
    if (auto xml = state.createXml()) copyXmlToBinary(*xml, destData);
}

void NeuralRysumProcessor::setStateInformation(const void* data, int sizeInBytes) {
    if (auto xml = getXmlFromBinary(data, sizeInBytes)) {
        auto state = juce::ValueTree::fromXml(*xml);
        if (!state.isValid()) return;
        const juce::String path = state.getProperty("modelPath").toString();
        apvts.replaceState(state);
        if (path.isNotEmpty() && juce::File(path).existsAsFile()) loadModel(juce::File(path));
    }
}

juce::AudioProcessorEditor* NeuralRysumProcessor::createEditor() { return new NeuralRysumEditor(*this); }

juce::AudioProcessor* JUCE_CALLTYPE createPluginFilter() { return new NeuralRysumProcessor(); }

#pragma once

#include "PluginProcessor.h"

// Two latent axes on a pad: x = Latent 1, y = Latent 2 (principal axes of the
// training data when the model was exported with --data).
class XYPad : public juce::Component {
public:
    XYPad(juce::RangedAudioParameter& x, juce::RangedAudioParameter& y) : px(x), py(y) {}

    void paint(juce::Graphics& g) override;
    void mouseDown(const juce::MouseEvent& e) override;
    void mouseDrag(const juce::MouseEvent& e) override;
    void mouseUp(const juce::MouseEvent& e) override;
    void mouseDoubleClick(const juce::MouseEvent& e) override;

private:
    void moveTo(juce::Point<float> p);
    juce::RangedAudioParameter& px;
    juce::RangedAudioParameter& py;
};

class NeuralRysumEditor : public juce::AudioProcessorEditor, private juce::Timer {
public:
    explicit NeuralRysumEditor(NeuralRysumProcessor&);
    ~NeuralRysumEditor() override;

    void paint(juce::Graphics&) override;
    void resized() override;

private:
    using SliderAttachment = juce::AudioProcessorValueTreeState::SliderAttachment;
    using ComboAttachment = juce::AudioProcessorValueTreeState::ComboBoxAttachment;
    using ButtonAttachment = juce::AudioProcessorValueTreeState::ButtonAttachment;

    struct Knob {
        juce::Slider slider;
        juce::Label label;
        std::unique_ptr<SliderAttachment> attachment;
    };
    struct Choice {
        juce::ComboBox box;
        juce::Label label;
        std::unique_ptr<ComboAttachment> attachment;
    };

    Knob& addKnob(const juce::String& id, const juce::String& text);
    Choice& addChoice(const juce::String& id, const juce::String& text);
    void timerCallback() override;
    void chooseModel();

    NeuralRysumProcessor& proc;
    juce::LookAndFeel_V4 look;
    juce::TextButton loadButton{"Load model..."}, captureButton{"Capture followed timbre"};
    juce::ToggleButton oscButton{"OSC out"};
    std::unique_ptr<ButtonAttachment> oscAttachment;
    juce::Label modelLabel, errorLabel;
    XYPad pad;
    juce::OwnedArray<Knob> knobs;
    juce::OwnedArray<Choice> choices;
    std::unique_ptr<juce::FileChooser> chooser;

    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(NeuralRysumEditor)
};

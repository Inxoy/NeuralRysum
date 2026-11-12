#include "PluginEditor.h"

namespace {
const juce::Colour kBack{0xff0b0b0b};
const juce::Colour kPanel{0xff161616};
const juce::Colour kLine{0xff3a3a3a};
const juce::Colour kText{0xffe8e8e8};
const juce::Colour kDim{0xff8a8a8a};
}  // namespace

// ---- XY pad ------------------------------------------------------------------
void XYPad::paint(juce::Graphics& g) {
    const auto r = getLocalBounds().toFloat();
    g.setColour(kPanel);
    g.fillRect(r);
    g.setColour(kLine);
    g.drawRect(r, 1.0f);
    // one grid line per standard deviation
    for (int i = 1; i < 6; ++i) {
        const float fx = r.getX() + r.getWidth() * static_cast<float>(i) / 6.0f;
        const float fy = r.getY() + r.getHeight() * static_cast<float>(i) / 6.0f;
        g.setColour(i == 3 ? kDim : kLine);
        g.drawVerticalLine(juce::roundToInt(fx), r.getY(), r.getBottom());
        g.drawHorizontalLine(juce::roundToInt(fy), r.getX(), r.getRight());
    }
    const float x = r.getX() + px.getValue() * r.getWidth();
    const float y = r.getBottom() - py.getValue() * r.getHeight();
    g.setColour(kText);
    g.drawEllipse(x - 7.0f, y - 7.0f, 14.0f, 14.0f, 1.5f);
    g.fillEllipse(x - 2.0f, y - 2.0f, 4.0f, 4.0f);
    g.setColour(kDim);
    g.setFont(12.0f);
    g.drawText("Latent 1", getLocalBounds().reduced(6), juce::Justification::bottomRight);
    g.drawText("Latent 2", getLocalBounds().reduced(6), juce::Justification::topLeft);
}

void XYPad::moveTo(juce::Point<float> p) {
    const auto r = getLocalBounds().toFloat();
    px.setValueNotifyingHost(juce::jlimit(0.0f, 1.0f, (p.x - r.getX()) / r.getWidth()));
    py.setValueNotifyingHost(juce::jlimit(0.0f, 1.0f, (r.getBottom() - p.y) / r.getHeight()));
    repaint();
}

void XYPad::mouseDown(const juce::MouseEvent& e) {
    px.beginChangeGesture();
    py.beginChangeGesture();
    moveTo(e.position);
}

void XYPad::mouseDrag(const juce::MouseEvent& e) { moveTo(e.position); }

void XYPad::mouseUp(const juce::MouseEvent&) {
    px.endChangeGesture();
    py.endChangeGesture();
}

void XYPad::mouseDoubleClick(const juce::MouseEvent&) {
    for (auto* p : {&px, &py}) {
        p->beginChangeGesture();
        p->setValueNotifyingHost(p->getDefaultValue());
        p->endChangeGesture();
    }
    repaint();
}

// ---- editor --------------------------------------------------------------------
NeuralRysumEditor::NeuralRysumEditor(NeuralRysumProcessor& p)
    : AudioProcessorEditor(&p),
      proc(p),
      pad(*p.apvts.getParameter(ids::latent(0)), *p.apvts.getParameter(ids::latent(1))) {
    look.setColour(juce::ResizableWindow::backgroundColourId, kBack);
    look.setColour(juce::Slider::thumbColourId, kText);
    look.setColour(juce::Slider::rotarySliderFillColourId, kText);
    look.setColour(juce::Slider::rotarySliderOutlineColourId, kLine);
    look.setColour(juce::Slider::textBoxTextColourId, kDim);
    look.setColour(juce::Slider::textBoxOutlineColourId, juce::Colours::transparentBlack);
    look.setColour(juce::Label::textColourId, kText);
    look.setColour(juce::ComboBox::backgroundColourId, kPanel);
    look.setColour(juce::ComboBox::outlineColourId, kLine);
    look.setColour(juce::ComboBox::textColourId, kText);
    look.setColour(juce::ComboBox::arrowColourId, kDim);
    look.setColour(juce::PopupMenu::backgroundColourId, kPanel);
    look.setColour(juce::PopupMenu::textColourId, kText);
    look.setColour(juce::PopupMenu::highlightedBackgroundColourId, kLine);
    look.setColour(juce::TextButton::buttonColourId, kPanel);
    look.setColour(juce::TextButton::textColourOffId, kText);
    look.setColour(juce::ToggleButton::textColourId, kText);
    look.setColour(juce::ToggleButton::tickColourId, kText);
    look.setColour(juce::ToggleButton::tickDisabledColourId, kDim);
    setLookAndFeel(&look);

    addAndMakeVisible(loadButton);
    loadButton.onClick = [this] { chooseModel(); };
    addAndMakeVisible(captureButton);
    captureButton.onClick = [this] { proc.captureFollowedTimbre(); };
    addAndMakeVisible(oscButton);
    oscAttachment = std::make_unique<ButtonAttachment>(proc.apvts, "oscOn", oscButton);

    addAndMakeVisible(modelLabel);
    modelLabel.setFont(juce::FontOptions(13.0f));
    modelLabel.setColour(juce::Label::textColourId, kDim);
    addAndMakeVisible(errorLabel);
    errorLabel.setFont(juce::FontOptions(13.0f));
    addAndMakeVisible(pad);

    addChoice("source", "Source");
    addChoice("timbre", "Timbre");
    addChoice("tuning", "Tuning");
    addChoice("root", "Meantone root");

    // row 1: remaining latent axes, row 2: descriptors and behaviour, row 3: utility
    for (int i = 2; i < ids::kLatents; ++i) addKnob(ids::latent(i), "Latent " + juce::String(i + 1));
    addKnob("bright", "Brightness");
    addKnob("noisy", "Noisiness");
    addKnob("spread", "Spread");
    addKnob("smooth", "Glide");
    addKnob("temp", "Temperature");
    addKnob("noise", "Model noise");
    addKnob("breath", "MIDI breath");
    addKnob("a4", "A4");
    addKnob("mix", "Dry / wet");
    addKnob("gain", "Output");
    addKnob("oscPort", "OSC port");

    setSize(860, 560);
    startTimerHz(30);
}

NeuralRysumEditor::~NeuralRysumEditor() {
    stopTimer();
    setLookAndFeel(nullptr);
}

NeuralRysumEditor::Knob& NeuralRysumEditor::addKnob(const juce::String& id, const juce::String& text) {
    auto* k = knobs.add(new Knob());
    k->slider.setSliderStyle(juce::Slider::RotaryHorizontalVerticalDrag);
    k->slider.setTextBoxStyle(juce::Slider::TextBoxBelow, false, 64, 16);
    k->label.setText(text, juce::dontSendNotification);
    k->label.setJustificationType(juce::Justification::centred);
    k->label.setFont(juce::FontOptions(12.0f));
    addAndMakeVisible(k->slider);
    addAndMakeVisible(k->label);
    k->attachment = std::make_unique<SliderAttachment>(proc.apvts, id, k->slider);
    return *k;
}

NeuralRysumEditor::Choice& NeuralRysumEditor::addChoice(const juce::String& id, const juce::String& text) {
    auto* c = choices.add(new Choice());
    if (auto* param = dynamic_cast<juce::AudioParameterChoice*>(proc.apvts.getParameter(id)))
        c->box.addItemList(param->choices, 1);
    c->label.setText(text, juce::dontSendNotification);
    c->label.setFont(juce::FontOptions(12.0f));
    c->label.setColour(juce::Label::textColourId, kDim);
    addAndMakeVisible(c->box);
    addAndMakeVisible(c->label);
    c->attachment = std::make_unique<ComboAttachment>(proc.apvts, id, c->box);
    return *c;
}

void NeuralRysumEditor::chooseModel() {
    chooser = std::make_unique<juce::FileChooser>("Load a NeuralRysum model", juce::File(), "*.nrys");
    const int chooserFlags = juce::FileBrowserComponent::openMode | juce::FileBrowserComponent::canSelectFiles;
    chooser->launchAsync(chooserFlags, [this](const juce::FileChooser& fc) {
        const juce::File f = fc.getResult();
        if (f == juce::File()) return;
        const juce::String err = proc.loadModel(f);
        errorLabel.setText(err, juce::dontSendNotification);
    });
}

void NeuralRysumEditor::timerCallback() {
    modelLabel.setText(proc.describeModel(), juce::dontSendNotification);
    pad.repaint();  // follows automation
}

void NeuralRysumEditor::paint(juce::Graphics& g) {
    g.fillAll(kBack);
    g.setColour(kText);
    g.setFont(juce::FontOptions(18.0f));
    g.drawText("NEURALRYSUM", 16, 10, 200, 24, juce::Justification::centredLeft);
}

void NeuralRysumEditor::resized() {
    auto area = getLocalBounds().reduced(16);
    auto top = area.removeFromTop(28);
    top.removeFromLeft(170);
    loadButton.setBounds(top.removeFromRight(120));
    top.removeFromRight(8);
    modelLabel.setBounds(top);
    errorLabel.setBounds(area.removeFromTop(20));
    area.removeFromTop(4);

    // left: pad with the four choices underneath; right: knobs
    auto left = area.removeFromLeft(300);
    pad.setBounds(left.removeFromTop(300));
    left.removeFromTop(10);
    for (auto* c : choices) {
        auto row = left.removeFromTop(26);
        c->label.setBounds(row.removeFromLeft(100));
        c->box.setBounds(row);
        left.removeFromTop(6);
    }
    auto buttons = left.removeFromTop(26);
    oscButton.setBounds(buttons.removeFromLeft(90));
    captureButton.setBounds(buttons);

    area.removeFromLeft(20);
    const int cols = 6;
    const int rows = (knobs.size() + cols - 1) / cols;
    const int cw = area.getWidth() / cols;
    const int ch = juce::jmin(150, area.getHeight() / juce::jmax(1, rows));
    for (int i = 0; i < knobs.size(); ++i) {
        juce::Rectangle<int> cell(area.getX() + (i % cols) * cw, area.getY() + (i / cols) * ch, cw, ch);
        cell = cell.reduced(4);
        knobs[i]->label.setBounds(cell.removeFromTop(16));
        knobs[i]->slider.setBounds(cell);
    }
}

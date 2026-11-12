cmake_minimum_required(VERSION 3.22)
project(NeuralRysum VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

# JUCE is downloaded at configure time. Note its licence: AGPLv3 or commercial.
include(FetchContent)
FetchContent_Declare(JUCE
  GIT_REPOSITORY https://github.com/juce-framework/JUCE.git
  GIT_TAG 8.0.4
  GIT_SHALLOW ON)
FetchContent_MakeAvailable(JUCE)

juce_add_plugin(NeuralRysum
  PRODUCT_NAME "NeuralRysum"
  COMPANY_NAME "NeuralRysum"
  PLUGIN_MANUFACTURER_CODE Nrys
  PLUGIN_CODE Nry1
  FORMATS VST3 Standalone
  IS_SYNTH FALSE
  NEEDS_MIDI_INPUT TRUE
  NEEDS_MIDI_OUTPUT FALSE
  VST3_CATEGORIES Fx Instrument
  COPY_PLUGIN_AFTER_BUILD FALSE)

target_sources(NeuralRysum PRIVATE
  Source/PluginProcessor.cpp
  Source/PluginEditor.cpp)

target_include_directories(NeuralRysum PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../engine/include)

target_compile_definitions(NeuralRysum PUBLIC
  JUCE_WEB_BROWSER=0
  JUCE_USE_CURL=0
  JUCE_VST3_CAN_REPLACE_VST2=0
  JUCE_DISPLAY_SPLASH_SCREEN=0)

# The engine's inner loops are plain C++ that the compiler vectorises.
# Without AVX2 it needs roughly three times the CPU.
if(MSVC)
  target_compile_options(NeuralRysum PRIVATE /arch:AVX2 /fp:fast)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(NeuralRysum PRIVATE -O3 -mavx2 -mfma)
else()
  target_compile_options(NeuralRysum PRIVATE -O3)
endif()

target_link_libraries(NeuralRysum
  PRIVATE
    juce::juce_audio_utils
    juce::juce_osc
  PUBLIC
    juce::juce_recommended_config_flags
    juce::juce_recommended_lto_flags)

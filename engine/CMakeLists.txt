# Stand-alone build of the engine's command line harness (parity, bench, midi).
# The plugin does not need this; it includes the headers directly.
cmake_minimum_required(VERSION 3.16)
project(nrys_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(nrys_cli tools/nrys_cli.cpp)
target_include_directories(nrys_cli PRIVATE include)
if(MSVC)
  target_compile_options(nrys_cli PRIVATE /arch:AVX2 /fp:fast)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_compile_options(nrys_cli PRIVATE -O3 -mavx2 -mfma)
endif()

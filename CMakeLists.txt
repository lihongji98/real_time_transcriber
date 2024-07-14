cmake_minimum_required(VERSION 3.28)
project(cpp_demo)

set(CMAKE_CXX_STANDARD 20)

set(ONNXRUNTIME_ROOT "/Users/lihongji/dev/dependency/onnxruntime")
set(PORTAUDIO_ROOT "/Users/lihongji/dev/dependency/portaudio")
set(EIGEN_ROOT "/Users/lihongji/dev/dependency/eigen")

set(PYTHON_ROOT "/Users/lihongji/anaconda3/envs/ml")


include_directories("${ONNXRUNTIME_ROOT}/include")
include_directories("${PORTAUDIO_ROOT}/include")
include_directories("${EIGEN_ROOT}")

include_directories("${PYTHON_ROOT}/include/python3.11")


find_package(Python3 COMPONENTS Development Interpreter NumPy REQUIRED)
find_package(SndFile REQUIRED)
find_package(Samplerate REQUIRED)


add_executable(cpp_demo src/main.cpp
        src/utils.cpp
        src/utils.h
        src/models.cpp
        src/models.h
        src/whisper_process.cpp
        src/whisper_process.h
        src/recorder.cpp
        src/recorder.h
)

target_link_libraries(cpp_demo "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.dylib")
target_link_libraries(cpp_demo "${PORTAUDIO_ROOT}/build/libportaudio.dylib")
target_link_libraries(cpp_demo "${PYTHON_ROOT}/lib/libpython3.11.dylib")
target_link_libraries(cpp_demo Python3::NumPy)

target_link_libraries(cpp_demo SndFile::sndfile)

find_library(SAMPLERATE_LIBRARY samplerate)
target_link_libraries(cpp_demo  ${SAMPLERATE_LIBRARY})
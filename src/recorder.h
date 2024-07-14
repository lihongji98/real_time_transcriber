#pragma once

#ifndef RECORDER_H
#define RECORDER_H

#include <thread>
#include <queue>
#include <mutex>

#include "portaudio.h"


class PortAudioException : public std::runtime_error {
public:
    explicit PortAudioException(PaError err) : std::runtime_error(Pa_GetErrorText(err)), error(err) {}
    PaError error;
};

class PortAudioHandler {
public:
    PortAudioHandler();
    ~PortAudioHandler();
};

class Recorder {
public:
    Recorder() : stream(nullptr), isRecording(false) {}
    ~Recorder() {stop();}
    void start();
    void stop();
    std::vector<float> getChunk();

private:
    PaStream* stream;
    std::atomic<bool> isRecording;
    std::thread recordingThread;
    std::queue<std::vector<float>> audioBuffer;
    std::mutex bufferMutex;

    void recordingLoop();
};

std::vector<float> recordAudio(int durationSeconds);
std::vector<float> processAudioData(const std::vector<float>& audioData);

#endif //RECORDER_H

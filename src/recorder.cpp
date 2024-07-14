#include <iostream>

#include "portaudio.h"
#include "recorder.h"


static const int SAMPLE_RATE = 16000;
static const int CHANNELS = 1;
static const int FRAMES_PER_BUFFER = 1024;
static constexpr float ENERGY_THRESHOLD = 3.0e-5f;
static constexpr int SILENCE_FRAMES = SAMPLE_RATE * 1;

const PaSampleFormat SAMPLE_FORMAT = paFloat32;



PortAudioHandler::PortAudioHandler(){
    PaError err = Pa_Initialize();
    if (err != paNoError) throw PortAudioException(err);
}

PortAudioHandler::~PortAudioHandler(){
    Pa_Terminate();
}

std::vector<float> recordAudio(int durationSeconds) {
    PortAudioHandler paHandler;
    PaStream *stream = nullptr;
    std::vector<float> recordedData;

    try {
        // Open an audio input stream
        PaError err = Pa_OpenDefaultStream(&stream,
                                           CHANNELS,          // input channels
                                           0,                 // output channels
                                           SAMPLE_FORMAT,
                                           SAMPLE_RATE,
                                           FRAMES_PER_BUFFER, // frames per buffer
                                           nullptr,              // no callback, use blocking API
                                           nullptr);             // no callback, so no user data
        if (err != paNoError) throw PortAudioException(err);

        // Use unique_ptr for automatic resource management
        std::unique_ptr<PaStream, decltype(&Pa_CloseStream)> streamGuard(stream, Pa_CloseStream);

        // Start the stream
        err = Pa_StartStream(stream);
        std::cout << "Start to record..." << std::endl;
        if (err != paNoError) throw PortAudioException(err);

        // Record audio data
        int numBuffers = (SAMPLE_RATE * durationSeconds) / FRAMES_PER_BUFFER;
        std::vector<float> buffer(FRAMES_PER_BUFFER);
        for (int i = 0; i < numBuffers; ++i) {
            err = Pa_ReadStream(stream, buffer.data(), FRAMES_PER_BUFFER);
            if (err != paNoError) throw PortAudioException(err);
            recordedData.insert(recordedData.end(), buffer.begin(), buffer.end());
        }

        // Stop the stream
        std::cout << "Stop recording..." << std::endl;
        err = Pa_StopStream(stream);
        if (err != paNoError) throw PortAudioException(err);

    } catch (const PortAudioException& e) {
        fprintf(stderr, "PortAudio error: %s\n", e.what());
        recordedData.clear();
    }

    return recordedData;
}

void Recorder::start() {
    if (isRecording) return;

    PaError err = Pa_Initialize();
    if (err != paNoError) throw PortAudioException(err);

    err = Pa_OpenDefaultStream(&stream,
                               CHANNELS,
                               0,
                               SAMPLE_FORMAT,
                               SAMPLE_RATE,
                               FRAMES_PER_BUFFER,
                               nullptr,
                               nullptr);
    if (err != paNoError) throw PortAudioException(err);

    err = Pa_StartStream(stream);
    if (err != paNoError) throw PortAudioException(err);

    isRecording = true;
    recordingThread = std::thread(&Recorder::recordingLoop, this);
    std::cout << "Recording started, press <Enter> to stop." << std::endl;
}

void Recorder::stop() {
    if (!isRecording) return;

    isRecording = false;
    if (recordingThread.joinable()) {
        recordingThread.join();
    }

    if (stream) {
        Pa_StopStream(stream);
        Pa_CloseStream(stream);
        stream = nullptr;
    }

    Pa_Terminate();
    std::cout << "Recording stopped..." << std::endl;
}

std::vector<float> Recorder::getChunk() {
    std::vector<float> chunk;
    std::lock_guard<std::mutex> lock(bufferMutex);
//    std::cout << audioBuffer.size() << " ";
    if (!audioBuffer.empty()) {
        chunk = std::move(audioBuffer.front());
        audioBuffer.pop();
    }
    return chunk;
}

void Recorder::recordingLoop() {
    std::vector<float> buffer(FRAMES_PER_BUFFER);
    bool isActive = false;

    while (isRecording) {
        PaError err = Pa_ReadStream(stream, buffer.data(), FRAMES_PER_BUFFER);
        if (err != paNoError) {
            std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
            return;
        }
        bool currentlyActive = detectVoiceActivity(buffer);
//        std::cout << currentlyActive << " ";

        if (currentlyActive) {
            if (!isActive) {
                isActive = true;
                currentChunk.clear();
            }
            silenceCounter = 0;
            currentChunk.insert(currentChunk.end(), buffer.begin(), buffer.end());
//            std::cout << currentChunk.size() << " ";
        } else {
            if (isActive) {
                silenceCounter += FRAMES_PER_BUFFER;
                currentChunk.insert(currentChunk.end(), buffer.begin(), buffer.end());

                if (silenceCounter >= SILENCE_FRAMES) {
                    isActive = false;
                    processAudioChunk(currentChunk);
                    currentChunk.clear();
                    silenceCounter = 0;
                }
            }
        }
    }

    // Process any remaining audio
    if (!currentChunk.empty()) {
        processAudioChunk(currentChunk);
    }
}

bool Recorder::detectVoiceActivity(const std::vector<float> &buffer) {
    float energy = 0.0f;
    for (float sample : buffer) {
        energy += sample * sample;
    }
    energy /= static_cast<float>(buffer.size());
//    std::cout << energy << " ";
    return energy > ENERGY_THRESHOLD;
}

void Recorder::processAudioChunk(std::vector<float> &chunk) {
    std::lock_guard<std::mutex> lock(bufferMutex);
    audioBuffer.push(std::move(chunk));
    while (audioBuffer.size() > 100) {
        audioBuffer.pop();
    }
}

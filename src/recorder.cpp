#include <iostream>

#include "portaudio.h"
#include "recorder.h"


const int SAMPLE_RATE = 16000;
const int CHANNELS = 1;
const int CHUNK_DURATION_SECONDS = 5;
const int FRAMES_PER_BUFFER = 1024;
const int BUFFERS_PER_CHUNK = (SAMPLE_RATE * CHUNK_DURATION_SECONDS) / FRAMES_PER_BUFFER;
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

std::vector<float> processAudioData(const std::vector<float>& audioData) {
    std::vector<float> processedAudio;
    processedAudio.reserve(audioData.size());

    std::transform(audioData.begin(), audioData.end(), std::back_inserter(processedAudio),
                   [](float sample) {
                       return std::clamp(sample, -1.0f, 1.0f);
                   });

    return processedAudio;
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
    if (!audioBuffer.empty()) {
        chunk = std::move(audioBuffer.front());
        audioBuffer.pop();
//        std::cout << audioBuffer.size() << std::endl;
    }
    return chunk;
}

void Recorder::recordingLoop() {
    std::vector<float> chunkBuffer;
    chunkBuffer.reserve(SAMPLE_RATE * CHUNK_DURATION_SECONDS);
    std::vector<float> buffer(FRAMES_PER_BUFFER);

    while (isRecording) {
        for (int i = 0; i < BUFFERS_PER_CHUNK; ++i) {
            PaError err = Pa_ReadStream(stream, buffer.data(), FRAMES_PER_BUFFER);
            if (err != paNoError) {
                std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
                return;
            }
            chunkBuffer.insert(chunkBuffer.end(), buffer.begin(), buffer.end());
        }

        std::lock_guard<std::mutex> lock(bufferMutex);
        audioBuffer.push(std::move(chunkBuffer));
        chunkBuffer.clear();
        chunkBuffer.reserve(SAMPLE_RATE * CHUNK_DURATION_SECONDS);

        // Optionally, limit the buffer size to prevent memory issues
        while (audioBuffer.size() > 10) {
            audioBuffer.pop();
        }
    }
}

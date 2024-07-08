#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <fstream>
#include <cstring>
#include <string>
#include <unordered_map>
#include <sstream>
#include <portaudio.h>
#include <stdexcept>
#include <memory>
#include <sndfile.h>


const int SAMPLE_RATE = 16000;
const int CHANNELS = 1;
const int FRAMES_PER_BUFFER = 1024;
const PaSampleFormat SAMPLE_FORMAT = paFloat32;


void softmax(std::vector<float>& input) {
    std::vector<float> exp_values(input.size());
    std::transform(input.begin(), input.end(), exp_values.begin(), [](float x) {
        return std::exp(x);
    });

    float sum_exp = std::accumulate(exp_values.begin(), exp_values.end(), 0.0f);

    std::transform(exp_values.begin(), exp_values.end(), exp_values.begin(), [sum_exp](float x) {
        return x / sum_exp;
    });

    input = std::move(exp_values);
}


size_t argsort_max(const std::vector<float>& output_probs) {
    if (output_probs.empty()) {
        throw std::runtime_error("Vector is empty");
    }
    size_t max_element_index = std::distance(output_probs.begin(), std::max_element(output_probs.begin(), output_probs.end()));

    return max_element_index;
}


std::unordered_map<int, std::string> load_vocab(const std::string& file_path) {
    std::unordered_map<int, std::string> vocab;
    std::ifstream file(file_path);
    std::string line;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return vocab;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        int index;

        if (std::getline(iss, word, '\t') && iss >> index) {
            vocab[index] = word;
        }
    }

    file.close();
    return vocab;
}


class PortAudioException : public std::runtime_error {
public:
    PortAudioException(PaError err) : std::runtime_error(Pa_GetErrorText(err)), error(err) {}
    PaError error;
};


class PortAudioHandler {
public:
    PortAudioHandler() {
        PaError err = Pa_Initialize();
        if (err != paNoError) throw PortAudioException(err);
    }
    ~PortAudioHandler() {
        Pa_Terminate();
    }
};


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


std::string read_file_string(const std::string& filePath) {
    std::ifstream file(filePath);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file");
    }

    std::stringstream buffer;
    buffer << file.rdbuf();

    file.close();

    return buffer.str();
}


std::vector<float> load_audio_data(const std::string& filename) {
    SF_INFO sfinfo;
    SNDFILE* file = sf_open(filename.c_str(), SFM_READ, &sfinfo);
    if (!file) {
        throw std::runtime_error("Error opening audio file");
    }

    // Read audio data
    std::vector<short> audio_data_short(sfinfo.frames * sfinfo.channels);
    sf_read_short(file, audio_data_short.data(), audio_data_short.size());
    sf_close(file);

    // Convert to float and normalize
    std::vector<float> audio_data(audio_data_short.size());
    for (size_t i = 0; i < audio_data.size(); ++i) {
        audio_data[i] = static_cast<float>(audio_data_short[i]) / 32768.0f;
    }

    // Convert to mono if stereo
    if (sfinfo.channels == 2) {
        for (size_t i = 0; i < audio_data.size() / 2; ++i) {
            audio_data[i] = (audio_data[i * 2] + audio_data[i * 2 + 1]) / 2.0f;
        }
        audio_data.resize(audio_data.size() / 2);
    }

    std::vector<float> resampled_audio;
    if (sfinfo.samplerate != SAMPLE_RATE) {
        double ratio = static_cast<double>(SAMPLE_RATE) / sfinfo.samplerate;
        size_t new_size = static_cast<size_t>(std::ceil(audio_data.size() * ratio));
        resampled_audio.resize(new_size);

        for (size_t i = 0; i < new_size; ++i) {
            double src_index = i / ratio;
            size_t src_index_int = static_cast<size_t>(src_index);
            double frac = src_index - src_index_int;

            if (src_index_int + 1 < audio_data.size()) {
                resampled_audio[i] = audio_data[src_index_int] * (1 - frac) +
                                     audio_data[src_index_int + 1] * frac;
            } else {
                resampled_audio[i] = audio_data[src_index_int];
            }
        }
    } else {
        resampled_audio = audio_data;
    }

    return resampled_audio;
}
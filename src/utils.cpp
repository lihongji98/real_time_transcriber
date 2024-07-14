#include <iostream>
#include <numeric>
#include <fstream>
#include <sstream>
#include <portaudio.h>
#include <sndfile.h>
#include <variant>


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


std::variant<std::unordered_map<int, std::string>,std::unordered_map<std::string, int>>
        load_vocab(const std::string& file_path, bool reverse=false) {
    std::ifstream file(file_path);
    std::string line;

    std::unordered_map<int, std::string> vocab;
    std::unordered_map<std::string, int> vocab_reverse;

    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return vocab;
    }

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string word;
        int index;

        if (std::getline(iss, word, '\t') && iss >> index){
            if (!reverse)
                vocab[index] = word;
            else
                vocab_reverse[word] = index;
        }
    }

    file.close();
    if (!reverse){
        vocab_reverse.clear();
        return vocab;
    }
    else{
        vocab.clear();
        return vocab_reverse;
    }
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

    if ((sfinfo.format & SF_FORMAT_SUBMASK) != SF_FORMAT_PCM_16) {
        sf_close(file);
        throw std::runtime_error("Audio file is not 16-bit PCM");
    }

    // Convert to mono if stereo
    if (sfinfo.channels == 2) {
        for (size_t i = 0; i < audio_data_short.size() / 2; ++i) {
            audio_data_short[i] = (audio_data_short[i * 2] + audio_data_short[i * 2 + 1]);
        }
        audio_data_short.resize(audio_data_short.size() / 2);
    }

    // Convert to float and normalize
    std::vector<float> audio_data(audio_data_short.size());
    for (size_t i = 0; i < audio_data.size(); ++i) {
        audio_data[i] = static_cast<float>(audio_data_short[i]) / 32768.0f;
    }

    std::vector<float> resampled_audio;
    if (sfinfo.samplerate != SAMPLE_RATE) {
        double ratio = static_cast<double>(SAMPLE_RATE) / sfinfo.samplerate;
        auto new_size = static_cast<size_t>(std::ceil(audio_data.size() * ratio));
        resampled_audio.resize(new_size);

        for (size_t i = 0; i < new_size; ++i) {
            double src_index = i / ratio;
            auto src_index_int = static_cast<size_t>(src_index);
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

bool endsWith(const std::string& str, const std::string& suffix="@@") {
    if (str.size() >= suffix.size()) {
        return str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
    } else {
        return false;
    }
}

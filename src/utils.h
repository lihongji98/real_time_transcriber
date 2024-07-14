#pragma once

#include <chrono>
#include "portaudio.h"




void softmax(std::vector<float>& input);

size_t argsort_max(const std::vector<float>& output_probs);

std::variant<std::unordered_map<int, std::string>,std::unordered_map<std::string, int>>
load_vocab(const std::string& file_path, bool reverse=false);

std::string read_file_string(const std::string& filePath);

std::vector<float> load_audio_data(const std::string& filename);

bool endsWith(const std::string& str, const std::string& suffix="@@");

class Timer{
public:
    std::chrono::time_point<std::chrono::steady_clock> start, end;
    std::chrono::duration<float> duration{};

    Timer(){
        start = std::chrono::high_resolution_clock::now();
    };
    ~Timer(){
        end = std::chrono::high_resolution_clock::now();
        duration = end - start;

        float ms = duration.count() * 1000.0f;
        std::cout << "Timer took " << ms << " ms." << "\n";
    };
};
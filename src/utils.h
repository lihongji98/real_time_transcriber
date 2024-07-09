#pragma once

#include "portaudio.h"




void softmax(std::vector<float>& input);
size_t argsort_max(const std::vector<float>& output_probs);
std::unordered_map<int, std::string> load_vocab(const std::string& file_path);
std::vector<float> recordAudio(int durationSeconds);
std::vector<float> processAudioData(const std::vector<float>& audioData);

std::vector<float> process_python_array(const std::vector<float>& input_vector, const std::string& python_script);
std::string read_file_string(const std::string& filePath);

std::vector<float> load_audio_data(const std::string& filename);

#pragma once

void softmax(std::vector<float>& input);
size_t argsort_max(const std::vector<float>& output_probs);
std::unordered_map<int, std::string> load_vocab(const std::string& file_path);
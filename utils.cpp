#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>
#include <fstream>
#include <string>
#include <unordered_map>
#include <sstream>


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
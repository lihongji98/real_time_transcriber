#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>


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

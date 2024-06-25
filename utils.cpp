#include <cmath>
#include <vector>
#include <algorithm>
#include <random>
#include <onnxruntime_cxx_api.h>
#include <iostream>


void sayHello() {
    std::cout << "Hello from functions.cpp!" << std::endl;
}

void printTensor(const Ort::Value& tensor) {
    // Get tensor info
    Ort::TensorTypeAndShapeInfo info = tensor.GetTensorTypeAndShapeInfo();
    std::vector<int64_t> shape = info.GetShape();
    size_t num_elements = info.GetElementCount();

    // Print shape
    std::cout << "Tensor shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        std::cout << shape[i];
        if (i < shape.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    // Print data
    std::cout << "Tensor data: ";
    if (tensor.IsTensor()) {
        switch (info.GetElementType()) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
            const float* data = tensor.GetTensorData<float>();
            for (size_t i = 0; i < std::min(num_elements, size_t(10)); ++i) {
                std::cout << data[i] << " ";
            }
            break;
        }
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
            const int64_t* data = tensor.GetTensorData<int64_t>();
            for (size_t i = 0; i < std::min(num_elements, size_t(10)); ++i) {
                std::cout << data[i] << " ";
            }
            break;
        }
                                                // Add more cases for other data types as needed
        default:
            std::cout << "Unsupported data type";
        }
    }
    else {
        std::cout << "Not a tensor";
    }
    std::cout << (num_elements > 10 ? "... (truncated)" : "") << std::endl;
}
#include <onnxruntime_cxx_api.h>
#include <iostream>
#include "utils.h"


int main() {
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session(nullptr);
    Ort::SessionOptions ort_session_options;
    auto modelPath = L"D:\\VisualStudioDev\\No-En-Transformer.onnx";

    session = Ort::Session(env, modelPath, ort_session_options);

    Ort::AllocatorWithDefaultOptions ort_alloc;

    size_t num_inputs = session.GetInputCount();
    std::vector<const char*> input_names;

    std::vector<int64_t> encoder_input_shapes;
    std::vector<int64_t> decoder_input_shapes;

    std::vector<Ort::Value> input_tensors;

    input_names.reserve(num_inputs);
    encoder_input_shapes.reserve(num_inputs); // encoder_input -> (1, 128) 
    decoder_input_shapes.reserve(num_inputs); // decoder_output -> (1, -1)

    for (size_t i = 0; i < num_inputs; i++) {
        Ort::AllocatedStringPtr input_temp = session.GetInputNameAllocated(i, ort_alloc);
        input_names.emplace_back(input_temp.get());
        input_temp.release();
    }
    
    Ort::AllocatedStringPtr output_temp = session.GetOutputNameAllocated(0, ort_alloc);
    const std::vector<const char*> output_name = { output_temp.get() };
    output_temp.release();


    encoder_input_shapes = {1, 128 };
    decoder_input_shapes = {1, 1 };
    

    std::vector<int64_t> encoder_input = { 1, 20, 88, 749, 21, 178, 10867, 46, 314, 56, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    std::vector<int64_t> decoder_input = { 1,};

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, encoder_input.data(), encoder_input.size(), encoder_input_shapes.data(), encoder_input_shapes.size()));
    input_tensors.emplace_back(Ort::Value::CreateTensor<int64_t>(memory_info, decoder_input.data(), decoder_input.size(), decoder_input_shapes.data(), decoder_input_shapes.size()));

    std::cout << encoder_input_shapes.size() << " " << decoder_input_shapes.size() << " " << input_tensors.size() << " " << output_name.size() << std::endl;

    std::vector<Ort::Value> output_tensors = session.Run(runOptions, input_names.data(), input_tensors.data(), input_tensors.size(), output_name.data(), output_name.size());

    std::cout << "Number of output tensors: " << output_tensors.size() << std::endl;

    for (size_t i = 0; i < output_tensors.size(); i++) {
        std::cout << "Output tensor " << i << ":" << std::endl;
        printTensor(output_tensors[i]);
        std::cout << std::endl;
    }

    std::cout << "Session Run is completed..." << std::endl;
}




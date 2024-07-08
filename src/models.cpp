#include "models.h"


const int MAX_LENGTH = 128;
const int TRANSFORMER_EOS = 2;
const int TRANSFORMER_VOC_SIZE = 16000;
const int WHISPER_EOS = 50257;
const int WHISPER_VOC_SIZE = 51865;
const int WHISPER_PROMPT_TOKEN_NUM = 3;

void Translator::load_model(const std::string &model_path) {
    session = Ort::Session(env, model_path.c_str(), ort_session_options);
    Ort::AllocatorWithDefaultOptions ort_alloc;

    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    input_names.reserve(num_inputs);
    output_names.reserve(num_outputs);
    encoder_input_shapes.reserve(num_inputs);
    decoder_input_shapes.reserve(num_outputs);

    for (size_t i = 0; i < num_inputs; i++) {
        Ort::AllocatedStringPtr input_temp = session.GetInputNameAllocated(i, ort_alloc);
        input_names.emplace_back(input_temp.get());
        input_temp.release();
    }

    for (size_t i = 0; i < num_outputs; i++){
        Ort::AllocatedStringPtr output_temp = session.GetOutputNameAllocated(i, ort_alloc);
        output_names.emplace_back(output_temp.get());
        output_temp.release();
    }
}

std::string Translator::infer(std::vector<int64_t>& encoder_input) {
    std::string output;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    encoder_input_shapes = { 1, 128 };
    decoder_input_shapes = { 1, 1 };

    std::vector<int64_t> decoder_input = { 1 };
    int64_t output_length_counter = 1;

    while (true) {
        input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(memory_info, encoder_input.data(),
                                                  encoder_input.size(), encoder_input_shapes.data(), encoder_input_shapes.size()));
        input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(memory_info, decoder_input.data(),
                                                  decoder_input.size(), decoder_input_shapes.data(), decoder_input_shapes.size()));

        std::vector<Ort::Value> output_tensors = session.Run(runOptions,
                                                             input_names.data(), input_tensors.data(), input_tensors.size(),
                                                             output_names.data(), output_names.size());

        Ort::Value& output_tensor = output_tensors[0];
        Ort::TensorTypeAndShapeInfo output_info = output_tensor.GetTensorTypeAndShapeInfo();
        size_t total_elements = output_info.GetElementCount();

        auto output_data = output_tensor.GetTensorMutableData<float>();
        std::vector<float> predict_token_vector(output_data + total_elements - TRANSFORMER_VOC_SIZE, output_data + total_elements);
        softmax(predict_token_vector);
        size_t next_token = argsort_max(predict_token_vector);

        decoder_input_shapes.clear();
        input_tensors.clear();

        if (next_token != TRANSFORMER_EOS) output.append(voc_trg[static_cast<int>(next_token)] + " ");

        decoder_input.push_back(static_cast<int>(next_token));
        output_length_counter++;
        decoder_input_shapes.push_back(1);
        decoder_input_shapes.push_back(output_length_counter);

        if ((next_token == TRANSFORMER_EOS) || (output_length_counter > MAX_LENGTH)) break;
    }

    return output;
}

void Transcriber::load_model(const std::string &model_path) {
    session = Ort::Session(env, model_path.c_str(), ort_session_options);
    Ort::AllocatorWithDefaultOptions ort_alloc;

    size_t num_inputs = session.GetInputCount();
    size_t num_outputs = session.GetOutputCount();

    input_names.reserve(num_inputs);
    output_names.reserve(num_outputs);
    encoder_input_shapes.reserve(num_inputs);
    decoder_input_shapes.reserve(num_outputs);

    for (size_t i = 0; i < num_inputs; i++) {
        Ort::AllocatedStringPtr input_temp = session.GetInputNameAllocated(i, ort_alloc);
        input_names.emplace_back(input_temp.get());
        input_temp.release();
    }

    for (size_t i = 0; i < num_outputs; i++){
        Ort::AllocatedStringPtr output_temp = session.GetOutputNameAllocated(i, ort_alloc);
        output_names.emplace_back(output_temp.get());
        output_temp.release();
    }
}

std::vector<int64_t> Transcriber::infer(std::vector<float> &encoder_input) {
    std::vector<int64_t> output;

    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    encoder_input_shapes = { 1, 80, 3000 };
    decoder_input_shapes = { 1, 1 };

    std::vector<int64_t> decoder_input = { 1 };
    int64_t output_length_counter = 1;

    while (true) {
        input_tensors.emplace_back(
                Ort::Value::CreateTensor<float>(memory_info, encoder_input.data(),
                                                  encoder_input.size(), encoder_input_shapes.data(), encoder_input_shapes.size()));
        input_tensors.emplace_back(
                Ort::Value::CreateTensor<int64_t>(memory_info, decoder_input.data(),
                                                  decoder_input.size(), decoder_input_shapes.data(), decoder_input_shapes.size()));

        std::vector<Ort::Value> output_tensors = session.Run(runOptions,
                                                             input_names.data(), input_tensors.data(), input_tensors.size(),
                                                             output_names.data(), output_names.size());

        Ort::Value& output_tensor = output_tensors[0];
        Ort::TensorTypeAndShapeInfo output_info = output_tensor.GetTensorTypeAndShapeInfo();
        size_t total_elements = output_info.GetElementCount();

        auto output_data = output_tensor.GetTensorMutableData<float>();
        std::vector<float> predict_token_vector(output_data + total_elements - WHISPER_VOC_SIZE, output_data + total_elements);
        softmax(predict_token_vector);
        size_t next_token = argsort_max(predict_token_vector);

        decoder_input_shapes.clear();
        input_tensors.clear();

        if ((next_token != WHISPER_EOS) && (output_length_counter > WHISPER_PROMPT_TOKEN_NUM))
            output.push_back(static_cast<int64_t>(next_token));

        decoder_input.push_back(static_cast<int>(next_token));
        output_length_counter++;
        decoder_input_shapes.push_back(1);
        decoder_input_shapes.push_back(output_length_counter);

        if ((next_token == WHISPER_EOS) || (output_length_counter > MAX_LENGTH)) break;
    }

    return output;
}

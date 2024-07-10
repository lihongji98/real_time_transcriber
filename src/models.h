#pragma once

#ifndef CPP_DEMO_MODELS_H
#define CPP_DEMO_MODELS_H

#include <iostream>
#include <filesystem>

#include "onnxruntime_cxx_api.h"
#include "utils.h"

class Transcriber {
public:
    Transcriber(const std::string& src_lang): ort_env(), runOptions(Ort::RunOptions()){};

    void load_model(const std::string& model_path);
    std::vector<int64_t> infer(std::vector<float>& encoder_input);

private:
    Ort::Env ort_env;
    Ort::RunOptions runOptions;
    Ort::Session session{nullptr};
    Ort::SessionOptions ort_session_options;

    std::unordered_map<int, std::string> voc_src;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<int64_t> encoder_input_shapes;
    std::vector<int64_t> decoder_input_shapes;

    std::vector<Ort::Value> input_tensors;
};


class Translator {
public:
    Translator():
            ort_env(),
            runOptions(Ort::RunOptions()){};

    void load_model(const std::string& model_path);
    std::vector<int> infer(std::vector<int64_t>& encoder_input);

private:
    Ort::Env ort_env;
    Ort::RunOptions runOptions;
    Ort::Session session{nullptr};
    Ort::SessionOptions ort_session_options;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<int64_t> encoder_input_shapes;
    std::vector<int64_t> decoder_input_shapes;

    std::vector<Ort::Value> input_tensors;
};


class Tokenizer{
public:
    std::string src_lang;
    std::string trg_lang;

    std::string src_sentence;

    Tokenizer(const std::string& src, const std::string& trg);
    ~Tokenizer();

    std::string preprocessing(const std::string& lines_to_translate);
    std::vector<int64_t> convert_token_to_id(const std::string& token_string);
    std::vector<std::string> convert_id_to_token(const std::vector<int>& token_ids);
    std::string postprocessing(const std::vector<std::string>& tokens);

    std::string decode(const std::vector<int>& token_ids);

private:
    std::string root = std::filesystem::current_path().string();

    std::string mosesdecoder_path = root + "/../transformer_onnx/tokenize_tool/mosesdecoder";
    std::string vocab_path = root + "/../transformer_onnx/voc";

    std::unordered_map<int, std::string> src_voc;
    std::unordered_map<int, std::string> trg_voc;

    std::unordered_map<std::string, int> reverse_src_voc;
    std::unordered_map<std::string, int> reverse_trg_voc;

    static std::string run_script(const std::vector<std::string>& script, const std::string& strings);
};


std::string transcribe(const std::vector<float>& audio_data);

#endif //CPP_DEMO_MODELS_H

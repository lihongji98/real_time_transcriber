#ifndef CPP_DEMO_MODELS_H
#define CPP_DEMO_MODELS_H

#include <iostream>
#include "onnxruntime_cxx_api.h"
#include "utils.h"

class Translator {
public:
    Translator(const std::string& src_lang, const std::string& trg_lang):
        env(),
        runOptions(Ort::RunOptions()),
        voc_trg(load_vocab(trg_lang)),
        voc_src(load_vocab(src_lang))
        {};

    void load_model(const std::string& model_path);
    std::string infer(std::vector<int64_t>& encoder_input);

private:
    Ort::Env env;
    Ort::RunOptions runOptions;
    Ort::Session session{nullptr};
    Ort::SessionOptions ort_session_options;

    std::unordered_map<int, std::string> voc_trg;
    std::unordered_map<int, std::string> voc_src;

    std::vector<const char*> input_names;
    std::vector<const char*> output_names;

    std::vector<int64_t> encoder_input_shapes;
    std::vector<int64_t> decoder_input_shapes;

    std::vector<Ort::Value> input_tensors;
};


#endif //CPP_DEMO_MODELS_H

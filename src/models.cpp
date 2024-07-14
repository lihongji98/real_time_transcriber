#include <cstdio>
#include <memory>
#include <stdexcept>

#include "models.h"
#include "utils.h"


const int MAX_LENGTH = 128;
const int TRANSFORMER_EOS = 2;
const int TRANSFORMER_VOC_SIZE = 16000;
const int WHISPER_EOS = 50257;
const int WHISPER_VOC_SIZE = 51865;
const int WHISPER_PROMPT_TOKEN_NUM = 3;

const int64_t SOS = 1;
const int64_t EOS = 2;
const int64_t PAD = 0;


void Translator::load_model(const std::string &model_path) {
    session = Ort::Session(ort_env, model_path.c_str(), ort_session_options);
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

std::vector<int> Translator::infer(std::vector<int64_t>& encoder_input) {
    std::vector<int> output;

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

        if (next_token != TRANSFORMER_EOS) output.push_back(static_cast<int>(next_token));

        decoder_input.push_back(static_cast<int>(next_token));
        output_length_counter++;
        decoder_input_shapes.push_back(1);
        decoder_input_shapes.push_back(output_length_counter);

        if ((next_token == TRANSFORMER_EOS) || (output_length_counter > MAX_LENGTH)) break;
    }

    return output;
}

void Transcriber::load_model(const std::string &model_path) {
    session = Ort::Session(ort_env, model_path.c_str(), ort_session_options);
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

std::vector<int64_t> Transcriber::infer(std::vector<float>& encoder_input) {
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

    return std::move(output);
}

Tokenizer::Tokenizer(const std::string& src, const std::string& trg) {
    src_lang = src;
    trg_lang = trg;
    std::string src_voc_path = root + "/../transformer_onnx/voc/voc_" + src_lang + ".txt";
    std::string trg_voc_path = root + "/../transformer_onnx/voc/voc_" + trg_lang + ".txt";

    //move to heap
    trg_voc = std::make_unique<std::unordered_map<int, std::string>>(
            std::get<std::unordered_map<int, std::string>>(load_vocab(trg_voc_path, false))
    );
    src_voc = std::make_unique<std::unordered_map<int, std::string>>(
            std::get<std::unordered_map<int, std::string>>(load_vocab(src_voc_path, false))
    );

    reverse_trg_voc = std::make_unique<std::unordered_map<std::string, int>>(
            std::get<std::unordered_map<std::string, int>>(load_vocab(trg_voc_path, true))
    );

    reverse_src_voc = std::make_unique<std::unordered_map<std::string, int>>(
            std::get<std::unordered_map<std::string, int>>(load_vocab(src_voc_path, true))
    );
}

Tokenizer::~Tokenizer() = default;

std::string Tokenizer::preprocessing(const std::string& lines_to_translate) {
    src_sentence = lines_to_translate;

    std::vector<std::string> command_normalize = {
            "perl", mosesdecoder_path + "/scripts/tokenizer/normalize-punctuation.perl", "-l", src_lang};
    std::vector<std::string> command_tokenize = {
            "perl", mosesdecoder_path + "/scripts/tokenizer/tokenizer.perl", "-l", src_lang};
    std::vector<std::string> command_truecase = {
            "perl", mosesdecoder_path + "/scripts/recaser/truecase.perl", "--model", vocab_path + "/truecase-model." + src_lang};
    std::vector<std::string> command_apply_bpe = {
            "python", vocab_path + "/apply_bpe.py", "-c", vocab_path + "/bpecode." + src_lang};

    std::string result = lines_to_translate;
    result = run_script(command_normalize, result);
    result = run_script(command_tokenize, result);
    result = run_script(command_truecase, result);
    result = run_script(command_apply_bpe, result);

    return result;
}

std::string Tokenizer::run_script(const std::vector<std::string>& script, const std::string& strings) {
    std::string result;

    std::string cmd = "echo '" + strings + "'" + " | ";
    for (const auto& arg : script) {
        cmd += arg + " ";
    }
    cmd += " 2>/dev/null";

    // Execute the command and capture the output

    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);

    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }

    constexpr size_t BUFFER_SIZE = 128;
    std::unique_ptr<char[]> buffer = std::make_unique<char[]>(BUFFER_SIZE);
    while (fgets(buffer.get(), BUFFER_SIZE, pipe.get()) != nullptr) {
        result += buffer.get();
    }

    if (!result.empty() && result.back() == '\n') {
        result.pop_back();
    }

    return result;
}

std::vector<int64_t> Tokenizer::convert_token_to_id(const std::string& token_string) {
    std::vector<std::string> tokens;
    std::string temp_string;
    for (const auto& e: token_string){
        if (e != ' '){
            temp_string += e;
        }else{
            tokens.push_back(temp_string);
            temp_string.clear();
        }
    }
    if (!temp_string.empty())
        tokens.push_back(temp_string);

    std::vector<int64_t> token_ids = {SOS};
    for (const auto& token: tokens){
        auto token_id =  static_cast<int64_t>(reverse_src_voc->at(token));
        token_ids.push_back(token_id);
    }
    token_ids.push_back(EOS);

    size_t count = token_ids.size();
    int PAD_NUM = MAX_LENGTH - static_cast<int>(count);
    for (int i=0; i < PAD_NUM; ++i)
        token_ids.push_back(PAD);

    return token_ids;
}

std::vector<std::string> Tokenizer::convert_id_to_token(const std::vector<int>& token_ids) {
    std::vector<std::string> tokens;
    tokens.reserve(token_ids.size());
    for (const auto& e: token_ids)
        tokens.emplace_back(trg_voc->at(e));

    return tokens;
}

std::string Tokenizer::postprocessing(const std::vector<std::string>& tokens){
    std::vector<std::string> command_detruecase = {
            "perl", mosesdecoder_path + "/scripts/recaser/detruecase.perl"};
    std::vector<std::string> command_detokenize = {
            "perl", mosesdecoder_path + "/scripts/tokenizer/detokenizer.perl", "-l", trg_lang};

    std::string glued_sentence;
    std::string temp;
    for (auto token: tokens){
        if (endsWith(token)){
            temp += token.substr(0, token.size() - 2);
        }else{
            if (!temp.empty())
                token = temp.append(token);
            glued_sentence += token + " ";
            temp.clear();
        }
    }

    std::string translated_sentence = glued_sentence;
    translated_sentence = run_script(command_detruecase, translated_sentence);
    translated_sentence = run_script(command_detokenize, translated_sentence);

    return translated_sentence;
}

std::string Tokenizer::decode(const std::vector<int> &token_ids) {
    std::vector<std::string> tokens = convert_id_to_token(token_ids);
    std::string string = postprocessing(tokens);

    char end_symbol = src_sentence[src_sentence.size() - 1];
    string += end_symbol;

    return string;
}


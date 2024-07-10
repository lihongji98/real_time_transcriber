#include <iostream>
#include <vector>

#include "models.h"
#include "utils.h"
#include "whisper_process.h"


std::string transcribe(const std::vector<float>& audio_data){
    std::string root = std::filesystem::current_path().string();
    const std::string python_preprocess_script = read_file_string(root + "/../whisper_onnx/scripts/process_script.py");
    const std::string python_decode_script = read_file_string(root + "/../whisper_onnx/scripts/decode_script.py");
    const std::string transcriber_path = root + "/../whisper_onnx/model/whisper.onnx";

    Transcriber transcriber("Norwegian");
    transcriber.load_model(transcriber_path);
    std::cout << "whisper onnx model is loaded..."  << " ";

    std::vector<float> processed_audio_np = process_python_array(audio_data, python_preprocess_script);
    std::cout << "wav vector process is finished... " << " ";

    const std::vector<int64_t > token_ids = transcriber.infer(processed_audio_np);
    std::string transcribed_sentence =  process_token_ids(token_ids, python_decode_script);

    return transcribed_sentence;
}


std::string translate(const auto& src_sentence){
    std::string model_path =  "../transformer_onnx/model/No-En-Transformer.onnx";

    Tokenizer tokenizer("no", "en");
    std::cout << "tokenizer and translator are initialized..." <<  std::endl;
    Translator translator;

    std::string s = tokenizer.preprocessing(src_sentence);
    std::vector<int64_t> encoder_input = tokenizer.convert_token_to_id(s);

    translator.load_model(model_path);
    std::vector<int> output = translator.infer(encoder_input);
    std::string res = tokenizer.decode(output);

    return res;
}


int main() {
    PythonEnvironment py_env;
//    int recordDurationSeconds = 1;
//    std::vector<float> audioData = recordAudio(recordDurationSeconds);
//    std::vector<float> audio_data = processAudioData(audioData);

    std::vector<float> audio_data = load_audio_data("../demo1.wav");
    std::cout << "wav file is loaded..." <<  " ";

    std::string src_sentence = transcribe(audio_data);
    std::cout << "trancription is finished..." <<  " ";

    std::string trg_sentence = translate(src_sentence);

    std::cout << trg_sentence << std::endl;

}

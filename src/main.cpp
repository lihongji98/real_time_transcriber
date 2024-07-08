#include <iostream>
#include <filesystem>
#include <vector>

#include "models.h"
#include "utils.h"
#include "whisper_process.h"


int main() {
    PythonEnvironment env;
    std::string root = std::filesystem::current_path().string();

//    int recordDurationSeconds = 3;
//    std::vector<float> audioData = recordAudio(recordDurationSeconds);
//    std::vector<float> audio_data = processAudioData(audioData);

    std::vector<float> audio_data = load_audio_data("../demo.wav");
    std::cout << "wav file is loaded..." << std::endl;

    const std::string python_preprocess_script = read_file_string(root + "/../whisper_onnx/scripts/process_script.txt");
    std::vector<float> processed_audio_np = process_python_array(audio_data, python_preprocess_script);


    const std::string transcriber_path = root + "/../whisper_onnx/model/whisper.onnx";
    Transcriber transcriber("Norwegian");
    transcriber.load_model(transcriber_path);
    const std::vector<int64_t > token_ids = transcriber.infer(processed_audio_np);

    const std::string python_decode_script = read_file_string(root + "/../whisper_onnx/scripts/decode_script.txt");
    std::string transcription =  process_token_ids(token_ids, python_decode_script);

    std::cout << transcription << std::endl;
//    std::vector<int64_t> encoder_input = { 1, 20, 88, 749, 21, 178, 10867, 46, 314, 56, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };
//
//    std::string src_voc_path = root + "/../transformer_onnx/voc/voc_no.txt";
//    std::string trg_voc_path = root + "/../transformer_onnx/voc/voc_en.txt";
//    Translator translator(src_voc_path, trg_voc_path);
//    std::string model_path = root + "/../transformer_onnx/model/No-En-Transformer.onnx";
//    translator.load_model(model_path);
//    std::string output = translator.infer(encoder_input);
//
//    std::cout << output << std::endl;

}

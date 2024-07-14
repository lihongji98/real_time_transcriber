#include <iostream>
#include <filesystem>
#include <future>

#include "whisper_process.h"
#include "recorder.h"
#include "utils.h"
#include "models.h"


struct ptr_wrapper{
    std::unique_ptr<Translator> translation_ptr;
    std::unique_ptr<Tokenizer> tokenizer_ptr;
};

std::unique_ptr<Transcriber> load_transcription_model(){
    std::string root = std::filesystem::current_path().string();
    const std::string transcriber_path = root + "/../whisper_onnx/model/whisper.onnx";

    auto transcriber = std::make_unique<Transcriber>();
    transcriber->load_model(transcriber_path);
    std::cout << "whisper onnx model is loaded..."  << " ";

    return transcriber;
}

std::string transcribe(const std::vector<float>& audio_data, const std::unique_ptr<Transcriber>& transcriber){
    std::string root = std::filesystem::current_path().string();
    const std::string python_preprocess_script = read_file_string(root + "/../whisper_onnx/scripts/process_script.py");
    const std::string python_decode_script = read_file_string(root + "/../whisper_onnx/scripts/decode_script.py");

    std::vector<float> processed_audio_np = process_python_array(audio_data, python_preprocess_script);

    const std::vector<int64_t > token_ids = transcriber->infer(processed_audio_np);
    std::string transcribed_sentence =  process_token_ids(token_ids, python_decode_script);

    return transcribed_sentence;
}

ptr_wrapper load_translation_model(){
    std::string root = std::filesystem::current_path().string();
    std::string model_path = root + "/../transformer_onnx/model/No-En-Transformer.onnx";

    auto tokenizer = std::make_unique<Tokenizer>("no", "en");
    auto translator = std::make_unique<Translator>();
    translator->load_model(model_path);
    std::cout << "transformer onnx model is loaded..."  << std::endl;

    return ptr_wrapper{std::move(translator), std::move(tokenizer)};
}

std::string translate(const std::string& src_sentence,
                      const std::unique_ptr<Translator>& translator,
                      const std::unique_ptr<Tokenizer>& tokenizer){
    if (src_sentence == "<|nocaptions|>")
        return "";

    std::string s = tokenizer->preprocessing(src_sentence);
    std::vector<int64_t> encoder_input = tokenizer->convert_token_to_id(s);
    std::vector<int> output = translator->infer(encoder_input);
    std::string res = tokenizer->decode(output);

    return res;
}


int main() {
    PythonEnvironment py_env;
    Recorder recorder;

    auto transcriber_ptr = load_transcription_model();
    auto ptr_wraper = load_translation_model();
    auto translation_ptr = std::move(ptr_wraper.translation_ptr);
    auto tokenizer_ptr = std::move(ptr_wraper.tokenizer_ptr);

    std::atomic<bool> shouldExit(false);

    std::thread inputThread([&shouldExit]() {
        std::cin.get(); // Wait for Enter key
        shouldExit = true;
    });

    recorder.start();

    // Record for a certain duration or until user input
    std::this_thread::sleep_for(std::chrono::seconds(3));

    while (!shouldExit) {
        auto chunk = recorder.getChunk();
        if (!chunk.empty()) {
            auto audio_chunk = processAudioData(chunk);
            std::string src_sentence = transcribe(audio_chunk, transcriber_ptr);
            std::string trg_sentence = translate(src_sentence, translation_ptr, tokenizer_ptr);
            if (!(trg_sentence.empty()))
                std::cout << trg_sentence << std::endl;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    recorder.stop();

    inputThread.join();
//    int recordDurationSeconds = 5;
//    std::vector<float> audio_data = recordAudio(recordDurationSeconds);
//    std::vector<float> audio_data = processAudioData(audioData);

//    std::vector<float> audio_data = load_audio_data("../demo.wav");
//    std::cout << "wav file is loaded..." <<  std::endl;
}

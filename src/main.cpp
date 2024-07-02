#include <iostream>
#include "models.h"
#include "portaudio.h"


int main() {
    std::vector<int64_t> encoder_input = { 1, 20, 88, 749, 21, 178, 10867, 46, 314, 56, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

    Translator translator("../voc/voc_no.txt", "../voc/voc_en.txt");
    std::string model_path = "../model/No-En-Transformer.onnx";
    translator.load_model(model_path);
    std::string output = translator.infer(encoder_input);

    std::cout << output << std::endl;

    PaError err = Pa_Initialize();
    if (err != paNoError)
    {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    std::cout << "PortAudio initialized successfully!" << std::endl;

    err = Pa_Terminate();
    if (err != paNoError)
    {
        std::cerr << "PortAudio error: " << Pa_GetErrorText(err) << std::endl;
        return 1;
    }

    std::cout << "PortAudio terminated successfully!" << std::endl;

    return 0;
}

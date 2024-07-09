import sys
import os
from transformers import AutoProcessor

root = os.getcwd()

processor = AutoProcessor.from_pretrained(root + "/../whisper_onnx/processor")


def process_audio_array(arr):
    inputs = processor(arr, return_tensors="np", sampling_rate=16000, language="no")
    encoder_input = inputs.input_features

    return encoder_input


processed_audio_array = process_audio_array(raw_audio_array)

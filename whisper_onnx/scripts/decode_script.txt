import sys
import os
from transformers import AutoProcessor


root = os.getcwd()

processor = AutoProcessor.from_pretrained(root + "/../whisper_onnx/processor")


def decode_tokens(arr):
    tokens = processor.decode(arr)

    return tokens


token_string = decode_tokens(token_ids)

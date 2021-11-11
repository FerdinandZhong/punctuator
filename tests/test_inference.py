from inference import InferenceArguments, Inference
import pytest


testing_args = InferenceArguments(
    model_name_or_path="models/punctuator",
    tokenizer_name="distilbert-base-uncased",
    tag2id_storage_path="models/tag2id.json"
)


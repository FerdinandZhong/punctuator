import numpy as np
import pytest

from dbpunctuator.inference import Inference, InferenceArguments
from tests.common import cleaned_data, processed_data  # noqa: F401

testing_args = InferenceArguments(
    model_name_or_path="Qishuai/distilbert_punctuator_en",
    tokenizer_name="distilbert-base-uncased",
)


def accuracy(prediction_labels, true_labels):
    return round(
        np.sum(prediction_labels == true_labels) / prediction_labels.shape[0], 3
    )


@pytest.mark.usefixtures("processed_data")
def test_inference(processed_data):  # noqa: F811
    test_texts = [" ".join(token_list) for token_list in processed_data[0]]
    inference = Inference(testing_args)
    results_text, results_labels = inference.punctuation(test_texts)
    assert len(results_text) == len(test_texts)

    for result_text, result_labels, true_labels in zip(
        results_text, results_labels, processed_data[1]
    ):
        assert result_text[0].isupper()
        acc = accuracy(np.array(result_labels), np.array(true_labels))
        print(f"output text: '{result_text}' with accuracy: {acc}")
        assert acc >= 0.8
    inference.terminate()

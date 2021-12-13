import argparse
import os

from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast


def upload_model(fine_tuned_model_path, fine_tuned_model_name, tokenizer_name):
    fine_tuned_model = DistilBertForTokenClassification.from_pretrained(
        fine_tuned_model_path
    )
    token = os.environ["HUGGINGFACE_TOKEN"]

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)
    print(tokenizer)
    fine_tuned_model.push_to_hub(fine_tuned_model_name, use_auth_token=token)
    tokenizer.push_to_hub(fine_tuned_model_name, use_auth_token=token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tuned_model_path", default="")
    parser.add_argument("--fine_tuned_model_name", default="distilbert_punctuator_en")
    parser.add_argument("--tokenizer_name", default="distilbert-base-uncased")
    args = parser.parse_args()

    upload_model(
        fine_tuned_model_path=args.fine_tuned_model_path,
        fine_tuned_model_name=args.fine_tuned_model_name,
        tokenizer_name=args.tokenizer_name,
    )

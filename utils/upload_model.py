import argparse
import os

from transformers import DistilBertForTokenClassification


def upload_model(fine_tuned_model_path):
    fine_tuned_model = DistilBertForTokenClassification.from_pretrained(
        fine_tuned_model_path
    )
    token = os.environ["HUGGINGFACE_TOKEN"]

    fine_tuned_model.push_to_hub("distilbert_punctuator_en", use_auth_token=token)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine_tuned_model_path", default="")
    args = parser.parse_args()

    upload_model(fine_tuned_model_path=args.fine_tuned_model_path)

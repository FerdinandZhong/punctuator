import logging

from inference import Inference, InferenceArguments
from utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


if __name__ == "__main__":
    args = InferenceArguments(
        model_name_or_path="models/punctuator",
        tokenizer_name="distilbert-base-uncased",
        tag2id_storage_path="models/tag2id.json",
    )

    inference = Inference(inference_args=args, verbose=True)

    test_texts_1 = [
        "how are you its been ten years since we met in shanghai im really happy to meet you again whats your current phone number",
        "my number is 82732212",
    ]
    logger.info(f"testing result {inference.punctuation(test_texts_1)}")

    test_texts_2 = [
        "hi sir have you registered your account in our platform its having a big sale right now",
        "oh yes i have its a good website",
        "great thank you sir here is an additional promo code 5566",
    ]
    logger.info(f"testing result {inference.punctuation(test_texts_2)}")

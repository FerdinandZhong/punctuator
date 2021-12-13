import logging

import torch
from pydantic import BaseModel
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from dbpunctuator.utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class TestingModelArguments(BaseModel):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Args:
        model_name(str): name or path of pre-trained model
        tokenizer_name(str): name of pretrained tokenizer
    """

    model_name: str
    tokenizer_name: str


class TestingModel:
    def __init__(self, arguments: TestingModelArguments) -> None:
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            arguments.tokenizer_name
        )
        self.classifer = DistilBertForTokenClassification.from_pretrained(
            arguments.model_name
        )

    def sample_output(self, inputs):
        tokenized_inputs = self.tokenizer(
            inputs,
            is_split_into_words=False,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        logger.info(f"tokenized inputs: {tokenized_inputs}")
        self.tokenized_input_ids = tokenized_inputs["input_ids"].to(self.device)
        self.attention_mask = tokenized_inputs["attention_mask"].to(self.device)

        logits = self.classifer(self.tokenized_input_ids, self.attention_mask).logits
        if self.device.type == "cuda":
            argmax_preds = logits.argmax(dim=2).detach().cpu().numpy()
        else:
            argmax_preds = logits.argmax(dim=2).detach().numpy()
        logger.info(f"outputs of model {argmax_preds}")


if __name__ == "__main__":
    args = TestingModelArguments(
        model_name="distilbert-base-multilingual-cased",
        tokenizer_name="distilbert-base-multilingual-cased",
    )

    testing_model = TestingModel(args)
    test_texts = ["中文测试", "Chinese testing"]
    testing_model.sample_output(test_texts)

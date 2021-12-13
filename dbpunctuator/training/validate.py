import json
import logging
from typing import Optional

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.metrics import classification_report
from torch._C import device  # noqa: F401
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from dbpunctuator.utils import NORMAL_TOKEN_TAG

from .dataset import PunctuatorDataset, read_data

logger = logging.getLogger(__name__)


class ValidationArguments(BaseModel):
    """Arguments pertaining to which model/config we are going to do the validation

    Args:
        data_file_path(str): path of validation data
        model_name_or_path(str): name or path of fine-tuned model
        tokenizer_name(str): name of tokenizer
        min_sequence_length(int): min sequence length of one sample
        max_sequence_length(int): max sequence length of one sample
        batch_size(int): batch size
        tag2id_storage_path(Optional[str]): tag2id storage path. Default one is from model config. Pass in this argument if your model doesn't have a tag2id inside config # noqa: E501
    """

    data_file_path: str
    model_name_or_path: str
    tokenizer_name: str
    min_sequence_length: int
    max_sequence_length: int
    batch_size: int
    tag2id_storage_path: Optional[str]


class ValidationPipeline:
    def __init__(self, validation_arguments) -> None:
        self.arguments = validation_arguments
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            validation_arguments.tokenizer_name
        )
        self.classifier = DistilBertForTokenClassification.from_pretrained(
            validation_arguments.model_name_or_path
        ).to(self.device)
        if validation_arguments.tag2id_storage_path:
            with open(validation_arguments.tag2id_storage_path, "r") as fp:
                self.tag2id = json.load(fp)
        else:
            self.tag2id = self.classifier.config.label2id

    def load_validation_data(self):
        logger.info("load validation data")
        self.texts, self.tags = read_data(
            self.arguments.data_file_path,
            self.arguments.min_sequence_length,
            self.arguments.max_sequence_length,
        )
        return self

    def tokenize(self):
        logger.info("tokenize data")
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.arguments.tokenizer_name
        )
        self.encodings = tokenizer(
            self.texts,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
        )
        self.labels = self._encode_tags(self.tags, self.encodings)
        self.dataset = PunctuatorDataset(self.encodings, self.labels)

        return self

    def validate(self):
        logger.info("start validation")
        val_loader = DataLoader(
            self.dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        self.classifier.train(False)

        steps = 0
        total_preds = []
        total_labels = []
        with tqdm(total=len(val_loader)) as pbar:
            for batch in val_loader:
                steps += 1
                pbar.set_description(f"Processing batch: {steps}")

                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.classifier(
                    input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = outputs.loss
                logits = outputs.logits

                true_preds, true_labels = self._post_process(
                    logits, labels, attention_mask
                )
                total_preds.extend(true_preds)
                total_labels.extend(true_labels)

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Last_batch_loss": f"{loss:.2f}",
                    }
                )

        tested_labels = []
        target_names = []
        for tag, id in self.tag2id.items():
            if tag != NORMAL_TOKEN_TAG:
                tested_labels.append(id)
                target_names.append(tag)
        report = classification_report(
            total_labels,
            total_preds,
            labels=tested_labels,
            target_names=target_names,
            zero_division=1,
        )
        logger.info(f"validation report: \n {report}")

    def run(self):
        self.load_validation_data().tokenize().validate()

    def _encode_tags(self, tags, encodings):
        logger.info("encoding tags")
        labels = [[self.tag2id[tag] for tag in doc] for doc in tags]
        encoded_labels = []
        with tqdm(total=len(labels)) as pbar:
            for doc_labels, doc_offset in zip(labels, encodings.offset_mapping):
                try:
                    # create an empty array of -100
                    doc_enc_labels = np.ones(len(doc_offset), dtype=int) * -100
                    arr_offset = np.array(doc_offset)

                    # set labels whose first offset position is 0 and the second is not 0
                    doc_enc_labels[
                        (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
                    ] = doc_labels
                    encoded_labels.append(doc_enc_labels.tolist())
                except ValueError as e:
                    logger.warning(f"error encoding: {str(e)}")
                pbar.update(1)

        return encoded_labels

    def _post_process(self, logits, labels, attention_mask):
        if self.device.type == "cuda":
            max_preds = logits.argmax(dim=2).detach().cpu().numpy().flatten()
            flattened_labels = labels.detach().cpu().numpy().flatten()
            flattened_attention = attention_mask.detach().cpu().numpy().flatten()
        else:
            max_preds = logits.argmax(dim=2).detach().numpy().flatten()
            flattened_labels = labels.detach().numpy().flatten()
            flattened_attention = attention_mask.detach().numpy().flatten()
        not_padding_labels = flattened_labels[flattened_attention == 1]
        not_padding_preds = max_preds[flattened_attention == 1]
        reduce_ignored = not_padding_labels >= 0
        true_labels = not_padding_labels[reduce_ignored]  # remove ignored -100
        true_preds = not_padding_preds[reduce_ignored]

        return true_preds, true_labels

import logging
import os
from os import environ
from typing import Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.metrics import classification_report
from torch._C import device  # noqa: F401
from torch.utils.data import DataLoader
from tqdm import tqdm

from punctuator.training.punctuation_data_process import EncodingDataset
from punctuator.utils import NORMAL_TOKEN_TAG, Models

logger = logging.getLogger(__name__)


class EvaluationArguments(BaseModel):
    """Arguments pertaining to which model/config we are going to do the validation

    Args:
        evaluation_corpus(List[List[str]]): list of sequences for evaluation, longest sequence should be no longer than pretrained LM's max_position_embedding(512) # noqa: E501
        evaluation_tags(List[List[int]]): tags(int) for evaluation (the GT)
        model_weight_name(str): name or path of fine-tuned model
        model(Optional(enum)): model selected from Enum Models, default is "DISTILBERT"
        tokenizer_name(str): name of tokenizer
        batch_size(int): batch size
        use_gpu(Optional[bool]): whether to use gpu for training, default is "True"
        label2id(Optional[Dict]): label2id. Default one is from model config. Pass in this argument if your model doesn't have a label2id inside config # noqa: E501
        gpu_device(Optional[int]): specific gpu card index, default is the CUDA_VISIBLE_DEVICES from environ
    """

    evaluation_corpus: List[List[str]]
    evaluation_tags: List[List[int]]
    model_weight_path: str
    model_weight_name: str = "finetuned_model.bin"
    model: Optional[Models] = Models.BERT_TOKEN_CLASSIFICATION
    tokenizer_name: str
    batch_size: int
    use_gpu: Optional[bool] = True
    label2id: Optional[Dict]
    gpu_device: Optional[int] = environ.get("CUDA_VISIBLE_DEVICES", 0)
    additional_tokenizer_config: Optional[Dict] = {}


class EvaluationPipeline:
    def __init__(self, evaluation_arguments) -> None:
        self.arguments = evaluation_arguments

        if torch.cuda.is_available() and evaluation_arguments.use_gpu:
            self.device = torch.device(f"cuda:{evaluation_arguments.gpu_device}")
        else:
            self.device = torch.device("cpu")

        self.label2id = evaluation_arguments.label2id
        self.id2label = {id: label for label, id in self.label2id.items()}

        model_collection = evaluation_arguments.model.value
        self.tokenizer = model_collection.tokenizer.from_pretrained(
            self.arguments.tokenizer_name,
            **evaluation_arguments.additional_tokenizer_config,
        )
        self.model_config = model_collection.config.from_pretrained(
            os.path.join(
                evaluation_arguments.model_weight_path, "finetuned_model_config.json"
            ),
            label2id=self.label2id,
            id2label=self.id2label,
            num_labels=len(self.id2label),
        )
        self.model = model_collection.model(self.model_config)
        model_weights_path = os.path.join(
            evaluation_arguments.model_weight_path,
            evaluation_arguments.model_weight_name,
        )
        self.model.load_state_dict(torch.load(model_weights_path))
        self.model.to(self.device)

    def tokenize(self):
        logger.info("tokenize data")

        self.encodings = self.tokenizer(
            self.arguments.evaluation_corpus,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
        )
        self.evaluation_encoded_tags = self._encode_tags(
            self.arguments.evaluation_tags, self.encodings
        )
        self.dataset = EncodingDataset(self.encodings, self.evaluation_encoded_tags)

        return self

    def validate(self):
        logger.info("start validation")
        val_loader = DataLoader(
            self.dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        self.model.train(False)

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
                outputs = self.model(
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
        for label, id in self.label2id.items():
            if label != NORMAL_TOKEN_TAG:
                tested_labels.append(id)
                target_names.append(label)
        report = classification_report(
            total_labels,
            total_preds,
            labels=tested_labels,
            digits=4,
            target_names=target_names,
            zero_division=1,
        )
        logger.info(f"testing report: \n {report}")

    def run(self):
        self.tokenize().validate()

    def _encode_tags(self, tags, encodings):
        logger.info("encoding tags")
        encoded_labels = []
        with tqdm(total=len(tags)) as pbar:
            for doc_labels, doc_offset in zip(tags, encodings.offset_mapping):
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

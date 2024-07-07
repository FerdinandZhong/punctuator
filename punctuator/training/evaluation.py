import argparse
import json
import logging
from os import environ
from typing import Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from sklearn.metrics import classification_report
from torch._C import device  # noqa: F401
from torch.utils.data import DataLoader
from tqdm import tqdm

from punctuator.utils import NORMAL_TOKEN_TAG, Models, model_type, str2bool

from .finetuning_data_process import process_data
from .punctuation_data_process import EncodingDataset

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
    model_weight_name: str
    model: Optional[Models] = Models.DISTILBERT
    tokenizer_name: str
    batch_size: int
    use_gpu: Optional[bool] = True
    label2id: Optional[Dict]
    gpu_device: Optional[int] = environ.get("CUDA_VISIBLE_DEVICES", 0)
    additional_tokenizer_config: Optional[Dict] = {}
    additional_model_config: Optional[Dict] = {}
    only_compute_positional_recal: bool = False

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:

        # Basic arguments
        parser.add_argument(
            "--evaluation_data_file_path",
            type=str,
            required=True,
            help="Path to training corpus file",
        )
        parser.add_argument(
            "--min_sequence_length",
            type=int,
            required=True,
            default=32,
            help="Minimum sequence length (count of the words) in each sample.",
        )
        parser.add_argument(
            "--max_sequence_length",
            type=int,
            required=True,
            default=128,
            help="Maximum sequence length (count of the words) in each sample.",
        )
        parser.add_argument(
            "--model_weight_name",
            type=str,
            required=True,
            help="Path or name of pre-trained model weight",
        )
        parser.add_argument(
            "--tokenizer_name",
            type=str,
            required=True,
            help="Name of pretrained tokenizer",
        )
        parser.add_argument(
            "--model",
            type=str,
            choices=[m.name for m in Models],
            help="Model to use",
        )

        # Training arguments
        parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
        parser.add_argument(
            "--label2id", type=str, required=True, help="Label to ID mapping"
        )
        parser.add_argument(
            "--use_gpu",
            type=str2bool,
            default=True,
            help="Whether to use GPU for training",
        )
        parser.add_argument(
            "--gpu_device", type=int, default=0, help="Local rank of the process"
        )
        # Model-specific arguments
        parser.add_argument(
            "--additional_model_config",
            type=str,
            help="JSON string of additional model config",
        )
        parser.add_argument(
            "--additional_tokenizer_config",
            type=str,
            default="{}",
            help="JSON string of additional model config",
        )
        parser.add_argument(
            "--only_compute_positional_recal",
            type=str2bool,
            default=False,
            help="whether only compute the positional recall"
        )
        return parser

    @staticmethod
    def generate_corpus(args: argparse.Namespace):
        with open(args.evaluation_data_file_path, "r", encoding="utf-8") as file:
            evaluation_raw = file.readlines()

        (
            evaluation_corpus,
            evaluation_tags,
        ) = process_data(
            evaluation_raw, args.min_sequence_length, args.max_sequence_length
        )

        try:
            label2id = json.loads(args.label2id)
        except json.JSONDecodeError:
            label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
        evaluation_tags = [[label2id[tag] for tag in doc] for doc in evaluation_tags]

        return (
            evaluation_corpus,
            evaluation_tags,
            label2id,
        )

    @classmethod
    def from_cli_args(
        cls,
        args: argparse.Namespace,
        evaluation_corpus: List[List[str]],
        evaluation_tags: List[List[str]],
        label2id: Dict,
    ):
        try:
            additional_model_config = json.loads(args.additional_model_config)
        except (json.JSONDecodeError, TypeError):
            additional_model_config = {}
        try:
            additional_tokenizer_config = json.loads(args.additional_tokenizer_config)
        except (json.JSONDecodeError, TypeError):
            additional_tokenizer_config = {}
        # Set the attributes from the parsed arguments.
        evaluation_pipeline_args = cls(
            evaluation_corpus=evaluation_corpus,
            evaluation_tags=evaluation_tags,
            model=model_type(args.model),
            model_weight_name=args.model_weight_name,
            tokenizer_name=args.tokenizer_name,
            batch_size=args.batch_size,
            additional_model_config=additional_model_config,
            gpu_device=args.gpu_device,
            label2id=label2id,
            additional_tokenizer_config=additional_tokenizer_config,
            only_compute_positional_recal=args.only_compute_positional_recal
        )

        return evaluation_pipeline_args


class EvaluationPipeline:
    def __init__(self, evaluation_arguments) -> None:
        self.arguments = evaluation_arguments

        if torch.cuda.is_available() and evaluation_arguments.use_gpu:
            self.device = torch.device(f"cuda:{evaluation_arguments.gpu_device}")
        else:
            self.device = torch.device("cpu")

        model_collection = evaluation_arguments.model.value
        self.tokenizer = model_collection.tokenizer.from_pretrained(
            self.arguments.tokenizer_name,
            **evaluation_arguments.additional_tokenizer_config,
        )
        self.classifier = model_collection.model.from_pretrained(
            evaluation_arguments.model_weight_name
        ).to(self.device)
        if evaluation_arguments.label2id:
            self.label2id = evaluation_arguments.label2id
        else:
            self.label2id = self.classifier.config.label2id

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
        # self.classifier.train(False)
        self.classifier.eval()

        steps = 0
        total_preds = []
        total_labels = []
        total_position_preds_recall = []
        total_position_labels_recall = []
        total_position_preds_precision = []
        total_position_labels_precision = []

        with tqdm(total=len(val_loader)) as pbar:
            for batch in val_loader:
                steps += 1
                pbar.set_description(f"Processing batch: {steps}")

                input_ids = batch["input_ids"].to(self.device).long()
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                outputs = self.classifier(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                true_preds, true_labels = self._post_process(
                    logits, labels, attention_mask
                )
                recall_position_preds, recall_position_labels = self._position_results(
                    logits, labels, attention_mask, result_type="recall"
                )
                precision_position_preds, precision_position_labels = self._position_results(
                    logits, labels, attention_mask, result_type="recall"
                )
                if not self.arguments.only_compute_positional_recal:
                    total_preds.extend(true_preds)
                    total_labels.extend(true_labels)
                total_position_preds_recall.extend(recall_position_preds)
                total_position_labels_recall.extend(recall_position_labels)
                total_position_preds_precision.extend(precision_position_preds)
                total_position_labels_precision.extend(precision_position_labels)

                pbar.update(1)

        if not self.arguments.only_compute_positional_recal:
            tested_labels = []
            target_names = []
            for label, label_id in self.label2id.items():
                if label != NORMAL_TOKEN_TAG:
                    tested_labels.append(label_id)
                    target_names.append(label)
            report = classification_report(
                total_labels,
                total_preds,
                labels=tested_labels,
                digits=4,
                target_names=target_names,
                zero_division=1,
            )
            logger.info("validation report: \n %s", report)

        if len(total_position_labels_recall) == len(total_position_preds_recall):
            total_recall = np.sum(
                np.array(total_position_preds_recall) == np.array(total_position_labels_recall)
            ) / len(total_position_preds_recall)
            logger.info("Total recall of puncts position: %.3f", total_recall)
        if len(total_position_labels_precision) == len(total_position_preds_precision):
            total_recall = np.sum(
                np.array(total_position_preds_precision) == np.array(total_position_labels_precision)
            ) / len(total_position_preds_precision)
            logger.info("Total precision of puncts position: %.3f", total_recall)

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
                    logger.warning("error encoding: %s", str(e))
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

    def _position_results(self, logits, labels, all_attention_mask, result_type:str="recall"):
        all_preds = logits.argmax(dim=-1).detach()
        all_labels = labels.detach()
        all_attention_mask = all_attention_mask.detach()
        if self.device.type == "cuda":
            all_preds = all_preds.cpu()
            all_labels = all_labels.cpu()
            all_attention_mask = all_attention_mask.cpu()

        position_predictions = []
        position_gts = []

        for predictions, labels, attention_mask in zip(
            all_preds, all_labels, all_attention_mask
        ):
            prediction_positions = (attention_mask == 1) & (predictions > 0)
            gt_positions = (attention_mask == 1) & (labels > 0)

            prediction_position_ids = prediction_positions.nonzero(as_tuple=True)[0]
            gt_position_ids = gt_positions.nonzero(as_tuple=True)[0]

            if result_type == "recall":
                position_ids = gt_position_ids
            elif result_type == "precision":
                position_ids = prediction_position_ids
            predictions_in_positions = predictions[position_ids].numpy()
            predictions_in_positions[predictions_in_positions > 1] = 1
            gt_in_positions = labels[position_ids].numpy()
            gt_in_positions[gt_in_positions > 1] = 1
            position_predictions.extend(predictions_in_positions)
            position_gts.extend(gt_in_positions)
            # position_gts[index, gt_position_ids] = 1

        return position_predictions, position_gts

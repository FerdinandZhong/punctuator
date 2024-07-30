import argparse
import json
import logging
from os import environ
from typing import Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from torch._C import device  # noqa: F401
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from punctuator.utils import Models, model_type, str2bool

from .finetuning_data_process import process_data

logger = logging.getLogger(__name__)


class InputsDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    # TODO: tokenize the data while loading
    def __getitem__(self, idx):

        return self.inputs[idx]

    def __len__(self):
        return len(self.inputs)


def collate_fn(batch):
    return {"inputs": batch}


class ClassificationArguments(BaseModel):
    """Arguments pertaining to which model/config we are going to do the validation

    Args:
        corpus(List[List[str]]): list of sequences for evaluation, longest sequence should be no longer than pretrained LM's max_position_embedding(512) # noqa: E501
        evaluation_tags(List[List[int]]): tags(int) for evaluation (the GT)
        model_weight_name(str): name or path of fine-tuned model
        model(Optional(enum)): model selected from Enum Models, default is "DISTILBERT"
        tokenizer_name(str): name of tokenizer
        batch_size(int): batch size
        use_gpu(Optional[bool]): whether to use gpu for training, default is "True"
        label2id(Optional[Dict]): label2id. Default one is from model config. Pass in this argument if your model doesn't have a label2id inside config # noqa: E501
        gpu_device(Optional[int]): specific gpu card index, default is the CUDA_VISIBLE_DEVICES from environ
        output_file_path(str): output file path
    """

    corpus: List[List[str]]
    model_weight_name: str
    model: Optional[Models] = Models.DISTILBERT
    tokenizer_name: str
    batch_size: int
    use_gpu: Optional[bool] = True
    label2id: Optional[Dict]
    id2label: Optional[Dict]
    gpu_device: Optional[int] = environ.get("CUDA_VISIBLE_DEVICES", 0)
    additional_tokenizer_config: Optional[Dict] = {}
    additional_model_config: Optional[Dict] = {}
    only_compute_positional_recal: bool = False
    output_file_path: str

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
            "--id2label", type=str, required=True, help="ID to Label mapping"
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
            help="whether only compute the positional recall",
        )
        parser.add_argument("--output_file_path", type=str, help="Output")
        return parser

    @staticmethod
    def generate_corpus(args: argparse.Namespace):
        with open(args.evaluation_data_file_path, "r", encoding="utf-8") as file:
            evaluation_raw = file.readlines()

        (corpus, _,) = process_data(
            evaluation_raw, args.min_sequence_length, args.max_sequence_length
        )

        try:
            label2id = json.loads(args.label2id)
            id2label = json.loads(args.id2label)
        except json.JSONDecodeError:
            label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
            id2label = {0: "O", 1: "PUNCT"}

        return (
            corpus,
            label2id,
            id2label
        )

    @classmethod
    def from_cli_args(
        cls,
        args: argparse.Namespace,
        corpus: List[List[str]],
        label2id: Dict,
        id2label: Dict
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
        pipeline_args = cls(
            corpus=corpus,
            model=model_type(args.model),
            model_weight_name=args.model_weight_name,
            tokenizer_name=args.tokenizer_name,
            batch_size=args.batch_size,
            additional_model_config=additional_model_config,
            gpu_device=args.gpu_device,
            label2id=label2id,
            id2label=id2label,
            additional_tokenizer_config=additional_tokenizer_config,
            output_file_path=args.output_file_path,
        )

        return pipeline_args


class ClassificationPipeline:
    def __init__(self, arguments) -> None:
        self.arguments = arguments

        if torch.cuda.is_available() and arguments.use_gpu:
            self.device = torch.device(f"cuda:{arguments.gpu_device}")
        else:
            self.device = torch.device("cpu")

        model_collection = arguments.model.value
        self.tokenizer = model_collection.tokenizer.from_pretrained(
            self.arguments.tokenizer_name,
            **arguments.additional_tokenizer_config,
        )
        self.classifier = model_collection.model.from_pretrained(
            arguments.model_weight_name
        ).to(self.device)
        if arguments.label2id:
            self.label2id = arguments.label2id
        else:
            self.label2id = self.classifier.config.label2id
        self.id2label = arguments.id2label
        self.dataset = None

    def generate_dataset(self):
        """
        Generates datasets for training and validation from tokenized data.

        Removes the 'offset_mapping' from the encodings and creates instances of EncodingDataset for training and validation datasets.

        Returns:
            self: The instance of the class itself for method chaining.
        """  # noqa E 501
        logger.info("generate dataset from tokenized data")
        self.dataset = InputsDataset(
            self.arguments.corpus,
        )

        return self

    def _tokenize(self, batch_inputs):
        encoding = self.tokenizer(
            batch_inputs,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        return encoding

    def inference(self):
        logger.info("start inference")
        val_loader = DataLoader(
            self.dataset, batch_size=self.arguments.batch_size, shuffle=False, collate_fn=collate_fn,
        )
        # self.classifier.train(False)
        self.classifier.eval()

        steps = 0

        file_writer = open(self.arguments.output_file_path, "w")

        with tqdm(total=len(val_loader)) as pbar:
            for batch in val_loader:
                steps += 1
                pbar.set_description(f"Processing batch: {steps}")
                
                tokens = []
                for input in batch["inputs"]:
                    tokens.extend(input)
                tokenized_inputs = self._tokenize(batch["inputs"])

                input_ids = tokenized_inputs["input_ids"].to(self.device).long()
                attention_mask = tokenized_inputs["attention_mask"].to(self.device)
                outputs = self.classifier(input_ids, attention_mask=attention_mask)
                logits = outputs.logits

                offset_marks = self._mark_ignored_tokens(
                    tokenized_inputs["offset_mapping"]
                )
                true_preds = self._post_process(logits, attention_mask, offset_marks)
                for label_id, token in zip(true_preds, tokens):
                    label = self.id2label[label_id]
                    file_writer.write("%s\t%s\n" % (token, label))

                pbar.update(1)

        file_writer.close()

    def _mark_ignored_tokens(self, offset_mapping):
        samples = []
        for sample_offset in offset_mapping:
            # create an empty array of -100
            sample_marks = np.ones(len(sample_offset), dtype=int) * -100
            arr_offset = np.array(sample_offset)

            # set labels whose first offset position is 0 and the second is not 0, only special tokens second is also 0
            sample_marks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = 0
            samples.append(sample_marks.tolist())

        return np.array(samples).flatten()

    def run(self):
        self.generate_dataset().inference()

    def _post_process(self, logits, attention_mask, offset_marks):
        if self.device.type == "cuda":
            max_preds = logits.argmax(dim=2).detach().cpu().numpy().flatten()
            # flattened_attention = attention_mask.detach().cpu().numpy().flatten()
        else:
            max_preds = logits.argmax(dim=2).detach().numpy().flatten()
            # flattened_attention = attention_mask.detach().numpy().flatten()
        # not_padding_preds = max_preds[flattened_attention == 1]
        reduce_ignored = offset_marks >= 0
        true_preds = max_preds[reduce_ignored]

        return true_preds

import argparse
import json
import logging
import os
import time
from typing import Dict, List, Optional

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_constant_schedule_with_warmup

from punctuator.utils import Models, model_type, str2bool

from .pretraining_data_process import process_data

logger = logging.getLogger(__name__)
DEFAULT_LABEL_WEIGHT = 0.1


class EncodingDataset(Dataset):
    def __init__(self, encodings, has_punctuation_list):
        self.encodings = encodings
        self.has_punctuation_list = has_punctuation_list

    def __getitem__(self, idx):
        # following the BERT's original pretraining method
        item = {
            key: val[idx] if torch.is_tensor(val[idx]) else torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["has_punctuation"] = torch.tensor(self.has_punctuation_list[idx]).type(
            torch.LongTensor
        )

        return item

    def __len__(self):
        return len(self.has_punctuation_list)


class PreTrainingArguments(BaseModel):
    """Arguments for further PreTraining of Bert-based model

    Args:
        # basic arguments
        training_corpus(List[List[str]]): list of sequences for training, longest sequence should be no longer than pretrained LM # noqa: E501
        validation_corpus(List[List[str]]): list of sequences for validation, longest sequence should be no longer than pretrained LM # noqa: E501
        training_has_punctuation_list(List[List[int]]): has_punctuation list in each text for training
        val_has_punctuation_list(List[List[int]]):  has_punctuation list in each text for validation
        model(Optional(enum)): model selected from Enum Models, default is "DISTILBERT"
        model_weight_name(str): name or path of pre-trained model weight
        tokenizer_name(str): name of pretrained tokenizer

        # training arguments
        epoch(int): number of epoch
        batch_size(int): batch size
        model_storage_dir(str): fine-tuned model storage path
        early_stop_count(int): after how many epochs to early stop training if valid loss not become smaller. default 3 # noqa: E501
        use_gpu(Optional[bool]): whether to use gpu for training, default is "True"
        gpu_device(Optional[int]): specific gpu card index, default is the CUDA_VISIBLE_DEVICES from environ
        warm_up_steps(int): warm up steps.
        r_drop(bool): whether to train with r-drop
        r_alpha(int): alpha value for kl divengence in the loss, default is 0
        plot_steps(int): record training status to tensorboard among how many steps
        tensorboard_log_dir(Optional[str]): the tensorboard logs output directory, default is "runs"

        # model arguments
        additional_model_config(Optional[Dict]): additional configuration for model
    """

    # basic args
    training_corpus: List[List[str]]
    training_has_punctuation_list: List[List[int]]
    validation_corpus: List[List[str]]
    val_has_punctuation_list: List[List[int]]
    model_weight_name: str
    tokenizer_name: str
    model: Optional[Models] = Models.DISTILBERT
    load_backbone_only: bool = True

    # training ars
    epoch: int
    batch_size: int
    model_storage_dir: str
    intermediate_persist_step: int = 5
    early_stop_count: Optional[int] = 3
    use_gpu: Optional[bool] = True
    gpu_device: Optional[int] = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    warm_up_steps: int = 1000
    r_drop: bool = False
    r_alpha: float = 0
    plot_steps: int = 50
    tensorboard_log_dir: Optional[str] = "runs"
    mask_rate: float = 0.15

    # model args
    additional_model_config: Optional[Dict]
    additional_tokenizer_config: Optional[Dict] = {}

    @staticmethod
    def add_cli_args(
        parser: argparse.ArgumentParser,
    ) -> argparse.ArgumentParser:

        # Basic arguments
        parser.add_argument(
            "--training_data_file_path",
            type=str,
            required=True,
            help="Path to training corpus file",
        )
        parser.add_argument(
            "--validation_data_file_path",
            type=str,
            required=True,
            help="Path to validation corpus file",
        )
        parser.add_argument(
            "--min_sequence_length",
            type=int,
            required=True,
            default=8,
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
            "--load_backbone_only",
            type=str2bool,
            default=True,
            help="Load the backbone model only.",
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
        parser.add_argument("--epoch", type=int, required=True, help="Number of epochs")
        parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
        parser.add_argument(
            "--model_storage_dir",
            type=str,
            required=True,
            help="Path to store the fine-tuned model",
        )
        parser.add_argument(
            "--early_stop_count",
            type=int,
            default=3,
            help="Epochs to stop training if no improvement",
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
        parser.add_argument(
            "--warm_up_steps", type=int, default=1000, help="Number of warm-up steps"
        )
        parser.add_argument(
            "--r_drop",
            type=str2bool,
            default=False,
            help="Whether to train with R-Drop",
        )
        parser.add_argument(
            "--r_alpha",
            type=float,
            default=0.5,
            help="Alpha value for KL divergence in R-Drop",
        )
        parser.add_argument(
            "--plot_steps",
            type=int,
            default=50,
            help="Steps interval to log status to TensorBoard",
        )
        parser.add_argument(
            "--tensorboard_log_dir",
            type=str,
            default="runs",
            help="TensorBoard logs directory",
        )
        parser.add_argument(
            "--intermediate_persist_step",
            type=int,
            default=5,
            help="Intermediate step to persist the model weights",
        )

        parser.add_argument(
            "--mask_rate",
            type=float,
            default=0.15,
            help="Masking rate",
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
        return parser

    @staticmethod
    def generate_corpus(args: argparse.Namespace):
        with open(args.training_data_file_path, "r", encoding="utf-8") as file:
            training_raw = file.readlines()

        with open(args.validation_data_file_path, "r", encoding="utf-8") as file:
            val_raw = file.readlines()

        (
            training_corpus,
            training_has_punctuation_list,
        ) = process_data(
            training_raw, args.min_sequence_length, args.max_sequence_length
        )

        (
            validation_corpus,
            val_has_punctuation_list,
        ) = process_data(val_raw, args.min_sequence_length, args.max_sequence_length)

        sample = training_corpus[0]
        logger.info("Corpus sample: %s", sample)
        logger.info("Punct count sample: %s", training_has_punctuation_list[0])

        return (
            training_corpus,
            validation_corpus,
            training_has_punctuation_list,
            val_has_punctuation_list,
        )

    @classmethod
    def from_cli_args(
        cls,
        args: argparse.Namespace,
        training_corpus: List[List[str]],
        validation_corpus: List[List[str]],
        training_has_punctuation_list: List[List[int]],
        val_has_punctuation_list: List[List[int]],
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
        training_pipeline_args = cls(
            training_corpus=training_corpus,
            validation_corpus=validation_corpus,
            training_has_punctuation_list=training_has_punctuation_list,
            val_has_punctuation_list=val_has_punctuation_list,
            model=model_type(args.model),
            load_backbone_only=args.load_backbone_only,
            model_weight_name=args.model_weight_name,
            tokenizer_name=args.tokenizer_name,
            epoch=args.epoch,
            batch_size=args.batch_size,
            model_storage_dir=args.model_storage_dir,
            intermediate_persist_step=args.intermediate_persist_step,
            additional_model_config=additional_model_config,
            gpu_device=args.gpu_device,
            warm_up_steps=args.warm_up_steps,
            r_drop=args.r_drop,
            r_alpha=args.r_alpha,
            tensorboard_log_dir=args.tensorboard_log_dir,
            plot_steps=args.plot_steps,
            early_stop_count=args.early_stop_count,
            mask_rate=args.mask_rate,
            additional_tokenizer_config=additional_tokenizer_config,
        )

        return training_pipeline_args


class PreTrainingPipeline:
    def __init__(self, training_arguments):
        """PreTraining pipeline

        Args:
            training_arguments (PreTrainingArguments): arguments passed to training pipeline
        """
        self.arguments = training_arguments
        logger.info("cuda available: %s", torch.cuda.is_available())
        self.tensorboard_writter = SummaryWriter(training_arguments.tensorboard_log_dir)

        model_collection = training_arguments.model.value

        self.model_config = model_collection.config.from_pretrained(
            training_arguments.model_weight_name,
            **training_arguments.additional_model_config,
        )

        self.tokenizer = model_collection.tokenizer.from_pretrained(
            training_arguments.tokenizer_name,
            **training_arguments.additional_tokenizer_config,
        )
        # special_token_list = [PUNCT_TOKEN]
        # special_tokens_dict = {"additional_special_tokens": special_token_list}
        # self.tokenizer.add_special_tokens(special_tokens_dict)
        # self.model_config.vocab_size = self.model_config.vocab_size + len(special_token_list)
        logger.info("loaded tokenizer: %s", self.tokenizer)
        # logger.info("special tokens: %s", self.tokenizer.additional_special_tokens_ids)
        logger.info("start loading model")
        if training_arguments.load_backbone_only:
            backbone_model = model_collection.backbone_model.from_pretrained(
                training_arguments.model_weight_name,
                config=self.model_config,
            )
            self.full_model = model_collection.model(
                self.model_config, backbone_model=backbone_model
            )

        else:
            self.full_model = model_collection.model.from_pretrained(
                training_arguments.model_weight_name,
                config=self.model_config,
            )

        self.model_class = self.full_model.__class__.__name__
        logger.info("model loaded")

        if torch.cuda.is_available() and training_arguments.use_gpu:
            if torch.cuda.device_count() > 1:
                self.full_model = torch.nn.DataParallel(self.full_model)
                self.full_model.cuda()
                self.device = torch.device("cuda")
                self.is_parallel = True
            else:
                self.device = torch.device(f"cuda:{training_arguments.gpu_device}")
                self.full_model.to(self.device)
                self.is_parallel = False

        else:
            self.device = torch.device("cpu")
            self.is_parallel = False

        self.total_steps = 0
        self.class_weights = None
        self.training_has_punctuation_list = None
        self.val_has_punctuation_list = None
        self.training_token_labels = None
        self.val_token_labels = None
        self.training_encodings = None
        self.val_encodings = None
        self.training_dataset = None
        self.val_dataset = None
        self.best_state_dict = None
        self.best_acc_state_dict = None

    def tokenize(self):
        """
        Tokenizes the training and validation corpora using the specified tokenizer.

        This method prepares the text data for further processing by converting it into a format that the model can understand.
        It also handles splitting the text into words where necessary and ensures that the resulting tokens are padded appropriately.
        """  # noqa E501
        logger.info("tokenize data")

        self.training_encodings = self.tokenizer(
            self.arguments.training_corpus,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
        )
        self.val_encodings = self.tokenizer(
            self.arguments.validation_corpus,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
        )

        self.training_has_punctuation_list = (
            self.arguments.training_has_punctuation_list
        )
        self.val_has_punctuation_list = self.arguments.val_has_punctuation_list

        self.training_token_labels = self.training_encodings.input_ids
        self.val_token_labels = self.val_encodings.input_ids

        return self

    def generate_dataset(self):
        """
        Generates datasets for training and validation from tokenized data.

        Removes the 'offset_mapping' from the encodings and creates instances of EncodingDataset for training and validation datasets.

        Returns:
            self: The instance of the class itself for method chaining.
        """  # noqa E 501
        logger.info("generate dataset from tokenized data")
        self.training_encodings.pop("offset_mapping")
        self.val_encodings.pop("offset_mapping")
        self.training_dataset = EncodingDataset(
            self.training_encodings,
            self.training_has_punctuation_list,
        )
        self.val_dataset = EncodingDataset(
            self.val_encodings,
            self.val_has_punctuation_list,
        )

        return self

    # TODO: Issue with mulit-gpu, input and target not matched
    def _all_mask(self, input_ids_all):
        masked_input_ids_all = input_ids_all.detach().clone()
        masked_labels_all = torch.full(input_ids_all.shape, -100)
        token_type_ids = torch.full(input_ids_all.shape, 0)

        for index, input_ids in enumerate(input_ids_all):
            valuable_input_ids = (
                (input_ids != self.tokenizer.cls_token_id)
                & (input_ids != self.tokenizer.sep_token_id)
                & (input_ids != self.tokenizer.pad_token_id)
            )

            valuable_input_ids_index = valuable_input_ids.nonzero(as_tuple=True)[0]
            rand_values = torch.rand(valuable_input_ids_index.shape[0])
            mask_arr = rand_values <= 0.15
            selection_index = valuable_input_ids_index[mask_arr]

            masked_input_ids_all[index, selection_index] = self.tokenizer.mask_token_id
            masked_labels_all[index, selection_index] = input_ids[selection_index]

        return masked_input_ids_all, masked_labels_all, token_type_ids

    def train(self):
        logger.info("start training")

        train_loader = DataLoader(
            self.training_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        optim = AdamW(self.full_model.parameters(), lr=1e-5)

        scheduler = get_constant_schedule_with_warmup(
            optim,
            num_warmup_steps=self.arguments.warm_up_steps,
        )

        best_val_loss = float("inf")
        best_val_acc = 0
        no_improvement_count = 0

        with tqdm(total=self.arguments.epoch) as pbar:
            for epoch in range(self.arguments.epoch):
                pbar.set_description(f"Processing epoch: {epoch + 1}")

                start_time = time.time()

                self.full_model.train()

                train_loss, train_acc = self._train(train_loader, optim, scheduler)
                val_loss, val_acc = self._train(val_loader, optim, scheduler, True)

                self.tensorboard_writter.add_scalar(
                    "Epoch Loss/train", train_loss, epoch + 1
                )
                self.tensorboard_writter.add_scalar(
                    "Epoch Loss/valid", val_loss, epoch + 1
                )

                self.tensorboard_writter.add_scalar(
                    "Epoch acc/train", train_acc, epoch + 1
                )
                self.tensorboard_writter.add_scalar(
                    "Epoch acc/valid", val_acc, epoch + 1
                )

                end_time = time.time()

                epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

                logger.info(
                    "Epoch: %02d | Epoch Time: %dm %ds",
                    epoch + 1,
                    epoch_mins,
                    epoch_secs,
                )
                logger.info(
                    "\tTrain Loss: %.3f | Train Acc: %.2f%%",
                    train_loss,
                    train_acc * 100,
                )
                logger.info(
                    "\t Val. Loss: %.3f |  Val. Acc: %.2f%%", val_loss, val_acc * 100
                )

                pbar.update(1)

                if (epoch + 1) % self.arguments.intermediate_persist_step == 0:
                    logger.info(
                        "Save the intermediate checkpoint for epoch: %d", epoch + 1
                    )

                    self._intermediate_persist(epoch + 1)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    if self.is_parallel:
                        self.best_state_dict = self.full_model.module.bert.state_dict()
                    else:
                        self.best_state_dict = self.full_model.bert.state_dict()
                    no_improvement_count = 0
                elif val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if self.is_parallel:
                        self.best_acc_state_dict = self.full_model.module.state_dict()
                    else:
                        self.best_acc_state_dict = self.full_model.state_dict()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if (
                        self.arguments.early_stop_count > 0
                        and no_improvement_count >= self.arguments.early_stop_count
                    ):
                        logger.info(
                            "No improvement for past %s epochs, early stop training.",
                            no_improvement_count,
                        )  # noqa E501
                        return self

            logger.info("fine-tune finished")

        self.tensorboard_writter.flush()

        return self

    def _intermediate_persist(self, epoch_index):
        if self.is_parallel:
            persist_moodel = self.full_model.module
        else:
            persist_moodel = self.full_model
        torch.save(
            persist_moodel.bert.state_dict(),
            os.path.join(
                self.arguments.model_storage_dir,
                f"epoch_{epoch_index}_pytorch_model.bin",
            ),
        )

    def persist(self):
        """Persist this model into the passed directory."""

        logger.info("persist fine-tuned model")

        # self.classifier.load_state_dict(self.best_state_dict)
        # self.classifier.save_pretrained(self.arguments.model_storage_dir)

        self.model_config.architectures = [self.model_class]
        self.model_config.save_pretrained(self.arguments.model_storage_dir)
        torch.save(
            self.best_state_dict,
            os.path.join(self.arguments.model_storage_dir, "pytorch_model.bin"),
        )
        torch.save(
            self.best_acc_state_dict,
            os.path.join(
                self.arguments.model_storage_dir, "pytorch_model_best_acc.bin"
            ),
        )

        logger.info(
            "further pretrained model stored to %s", self.arguments.model_storage_dir
        )

    def _train(self, iterator, optim, scheduler=None, is_val=False):
        epoch_loss = 0
        epoch_acc = 0
        if is_val:
            self.full_model.train(False)
        else:
            self.full_model.train()

        in_epoch_steps = 0

        with tqdm(total=len(iterator)) as pbar:
            for batch in iterator:
                in_epoch_steps += 1
                pbar.set_description(f"Processing batch: {in_epoch_steps}")

                optim.zero_grad()
                masked_input_ids, masked_labels, token_type_ids = self._all_mask(
                    batch["input_ids"]
                )
                masked_input_ids = masked_input_ids.to(self.device)
                masked_labels = masked_labels.to(self.device)
                token_type_ids = token_type_ids.to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                has_punctuation = batch["has_punctuation"].to(self.device)

                outputs = self.full_model(
                    masked_input_ids,
                    attention_mask=attention_mask,
                    labels=masked_labels,
                    token_type_ids=token_type_ids,
                    has_punctuation_label=has_punctuation,
                )
                prediction_logits = outputs.prediction_logits
                loss = outputs.loss

                if self.is_parallel:
                    loss = loss.mean()

                if not is_val:
                    loss.backward()
                    optim.step()
                    if scheduler:
                        scheduler.step()
                    if self.total_steps % self.arguments.plot_steps == 0:
                        self.tensorboard_writter.add_scalar(
                            "Step Loss/train", loss, self.total_steps
                        )

                self.total_steps += 1

                epoch_loss += loss.item()
                epoch_acc += self._accuracy(prediction_logits, masked_labels)

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Last_loss": f"{loss:.3f}",
                        "Avg_cum_loss": f"{epoch_loss/in_epoch_steps:.3f}",
                    }
                )

        return epoch_loss / in_epoch_steps, epoch_acc / in_epoch_steps

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _accuracy(self, prediction_logits, labels):
        if self.device.type == "cuda":
            max_preds = (
                prediction_logits.argmax(dim=-1).detach().cpu().numpy().flatten()
            )
            flattened_labels = labels.detach().cpu().numpy().flatten()
        else:
            max_preds = prediction_logits.argmax(dim=-1).detach().numpy().flatten()
            flattened_labels = labels.detach().numpy().flatten()

        reduce_ignored = flattened_labels >= 0
        true_preds = max_preds[reduce_ignored]  # remove ignored -100
        true_labels = flattened_labels[reduce_ignored]

        if true_preds.shape[0] == true_labels.shape[0]:
            return np.sum(true_preds == true_labels) / true_labels.shape[0]

        return 0

    def run(self):
        self.tokenize().generate_dataset().train().persist()

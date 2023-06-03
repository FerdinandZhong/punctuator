import logging
import os
import time
from typing import Dict, List, Optional, Union

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_constant_schedule_with_warmup

from punctuator.utils import Models

from .model import SpanMaskedLM
from .punctuation_data_process import (
    PUNCT_TOKEN,
    SPACE_TOKEN,
    EncodingDataset,
    MaskedEncodingDataset,
)

logger = logging.getLogger(__name__)
DEFAULT_LABEL_WEIGHT = 1


class PretrainingArguments(BaseModel):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Args:
        # basic arguments
        training_corpus(List[List[str]]): list of sequences for training, longest sequence should be no longer than pretrained LM # noqa: E501
        validation_corpus(List[List[str]]): list of sequences for validation, longest sequence should be no longer than pretrained LM # noqa: E501
        training_tags(List[List[int]]): tags(int) for training
        validation_tags(List[List[int]]): tags(int) for validation
        plm_model(Optional(enum)): model selected from Enum Models, default is "DISTILBERT"
        plm_model_weight_name(str): name or path of pre-trained model weight
        tokenizer_name(str): name of pretrained tokenizer

        # training arguments
        epoch(int): number of epoch
        batch_size(int): batch size
        model_storage_dir(str): fine-tuned model storage path
        label2id(Dict): the tags label and id mapping
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
    training_corpus: Union[List[List[str]], List[str]]
    validation_corpus: Union[List[List[str]], List[str]]
    training_span_labels: List[List[int]]
    validation_span_labels: List[List[int]]
    plm_model: Optional[Models] = Models.BERT
    plm_model_config_name: str
    tokenizer_name: str
    current_plm_weights: Optional[str] = None

    # training ars
    epoch: int
    batch_size: int
    model_storage_dir: str
    early_stop_count: Optional[int] = 3
    use_gpu: Optional[bool] = True
    gpu_device: Optional[int] = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    warm_up_steps: int = 1000
    plot_steps: int = 50
    tensorboard_log_dir: Optional[str] = "runs"
    intermediate_persist_step: int = 5
    is_dynamic_mask: bool = False

    # model args
    span_only: bool = True
    mask_rate: float = 0.15
    additional_model_config: Optional[Dict] = {}
    additional_tokenizer_config: Optional[Dict] = {}
    load_weight: bool = False


class PretrainingPipeline:
    def __init__(self, training_arguments, verbose=False):
        """Training pipeline

        Args:
            training_arguments (TrainingArguments): arguments passed to training pipeline
        """
        self.arguments = training_arguments

        self.tensorboard_writter = SummaryWriter(training_arguments.tensorboard_log_dir)

        plm_model_collection = training_arguments.plm_model.value

        self.plm_model_config = plm_model_collection.config.from_pretrained(
            training_arguments.plm_model_config_name,
            **training_arguments.additional_model_config,
        )
        if verbose:
            logger.info(f"Full model config: {self.plm_model_config}")

        self.tokenizer = plm_model_collection.tokenizer.from_pretrained(
            training_arguments.tokenizer_name,
            **training_arguments.additional_tokenizer_config,
        )

        special_tokens_dict = {"additional_special_tokens": [SPACE_TOKEN, PUNCT_TOKEN]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.plm_model_config.vocab_size = len(self.tokenizer)
        lm_model = plm_model_collection.model(
            self.plm_model_config, add_pooling_layer=False
        )

        lm_model.resize_token_embeddings(len(self.tokenizer))
        self.model = SpanMaskedLM(
            self.plm_model_config,
            lm_model=lm_model,
            span_only=training_arguments.span_only,
        )
        if training_arguments.load_weight:
            if training_arguments.current_plm_weights:
                self.model.load_state_dict(
                    torch.load(training_arguments.current_plm_weights)
                )
            else:
                pretrained_bert = plm_model_collection.model.from_pretrained(
                    training_arguments.plm_model_config_name, add_pooling_layer=False
                )
                pretrained_bert.resize_token_embeddings(len(self.tokenizer))
                self.model.lm_model.load_state_dict(pretrained_bert.state_dict())

        self.span_token_id_start = self.tokenizer.additional_special_tokens_ids[0]
        self.span_token_ids = self.tokenizer.additional_special_tokens_ids

        if torch.cuda.is_available() and training_arguments.use_gpu:
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.cuda()
                self.device = torch.device("cuda")
                self.is_parallel = True
            else:
                self.device = torch.device(f"cuda:{training_arguments.gpu_device}")
                self.model.to(self.device)
                self.is_parallel = False
        else:
            self.device = torch.device("cpu")
            self.is_parallel = False

        if verbose:
            logger.info(f"Full model architecture: {self.model}")

    def tokenize(self, is_split_into_words=True):
        logger.info("tokenize data")

        self.training_encodings = self.tokenizer(
            self.arguments.training_corpus,
            is_split_into_words=is_split_into_words,
            padding=True,
            return_tensors="pt",
        )
        self.val_encodings = self.tokenizer(
            self.arguments.validation_corpus,
            is_split_into_words=is_split_into_words,
            padding=True,
            return_tensors="pt",
        )

        self.training_span_labels = self._span_label_encoding(
            self.training_encodings.input_ids, self.arguments.training_span_labels
        )
        self.val_span_labels = self._span_label_encoding(
            self.val_encodings.input_ids, self.arguments.validation_span_labels
        )

        if not self.arguments.span_only:
            self.training_token_labels = self._token_label_encoding(
                self.training_encodings.input_ids
            )
            self.val_token_labels = self._token_label_encoding(
                self.val_encodings.input_ids
            )
        else:
            self.training_token_labels, self.val_token_labels = None, None

        return self

    def _span_label_encoding(self, input_ids_all, span_labels_all):
        logger.info("tokenize span labels")
        span_enc_labels_list = []
        with tqdm(total=input_ids_all.shape[0]) as pbar:
            index = 0
            for input_ids, span_labels in zip(input_ids_all, span_labels_all):
                try:
                    span_enc_labels = np.ones(len(input_ids), dtype=int) * -100

                    span_enc_labels[
                        (
                            sum(
                                input_ids == span_token_id
                                for span_token_id in self.span_token_ids
                            )
                            * (input_ids != self.tokenizer.pad_token_id)
                        ).bool()  # for span label, must ignore the padding, as it is binary cross entropy  # noqa: E501
                    ] = span_labels
                    span_enc_labels_list.append(span_enc_labels.tolist())
                except ValueError as e:
                    logger.warning(f"error encoding span labels: {str(e)}")
                    logger.warning(f"index: {index}")
                pbar.update(1)
                index += 1
        return span_enc_labels_list

    def _token_label_encoding(self, input_ids_all):
        logger.info("tokenize token labels")
        token_enc_labels_list = []
        with tqdm(total=input_ids_all.shape[0]) as pbar:
            index = 0
            for input_ids in input_ids_all:
                try:
                    span_enc_label = sum(
                        input_ids == span_token_id
                        for span_token_id in self.span_token_ids
                    ).bool()
                    token_enc_labels = torch.where(span_enc_label, -100, input_ids)
                    token_enc_labels_list.append(token_enc_labels)
                except ValueError as e:
                    logger.warning(f"error encoding span labels: {str(e)}")
                    logger.warning(f"index: {index}")
                pbar.update(1)
                index += 1
        return token_enc_labels_list

    def static_mask(self):
        if self.arguments.span_only:
            logger.info(f"add mask to spans only")
            self.train_mask_input_ids = self._span_only_mask(
                self.training_encodings.input_ids
            )
            self.val_mask_input_ids = self._span_only_mask(self.val_encodings.input_ids)
        else:
            logger.info(f"add mask to all")
            self.train_mask_input_ids = self._all_mask(
                self.training_encodings.input_ids
            )
            self.val_mask_input_ids = self._all_mask(self.val_encodings.input_ids)
        return self

    def dynamic_mask(self, input_ids_batch):
        if self.arguments.span_only:
            masked_input_ids_batch = self._span_only_mask(input_ids_batch)
        else:
            masked_input_ids_batch = self._all_mask(input_ids_batch)
        return masked_input_ids_batch

    def _span_only_mask(self, input_ids_all):
        masked_input_ids_all = input_ids_all.detach().clone()
        with tqdm(
            total=input_ids_all.shape[0], disable=self.arguments.is_dynamic_mask
        ) as pbar:
            index = 0
            for input_ids in input_ids_all:
                span_enc_label = sum(
                    input_ids == span_token_id for span_token_id in self.span_token_ids
                ).bool()
                span_input_ids_index = span_enc_label.nonzero().squeeze()
                span_rand = torch.rand(span_input_ids_index.shape[0])
                span_mask_arr = span_rand < self.arguments.mask_rate
                span_selection_index = span_input_ids_index[
                    torch.flatten((span_mask_arr).nonzero()).tolist()
                ]

                masked_input_ids_all[
                    index, span_selection_index
                ] = self.tokenizer.mask_token_id
                index += 1
                pbar.update(1)
        return masked_input_ids_all

    def _all_mask(self, input_ids_all):
        masked_input_ids_all = input_ids_all.detach().clone()
        rand = torch.rand(input_ids_all.shape)
        with tqdm(
            total=input_ids_all.shape[0], disable=self.arguments.is_dynamic_mask
        ) as pbar:
            index = 0
            for input_ids in input_ids_all:
                valuable_input_ids = (
                    (input_ids != self.tokenizer.cls_token_id)
                    * (input_ids != self.tokenizer.sep_token_id)
                    * (input_ids != self.tokenizer.pad_token_id)
                ).bool()
                all_valuable_input_ids_index = valuable_input_ids.nonzero().squeeze()
                rand = torch.rand(all_valuable_input_ids_index.shape[0])
                mask_arr = rand < self.arguments.mask_rate * 2
                selection_index = all_valuable_input_ids_index[
                    torch.flatten((mask_arr).nonzero()).tolist()
                ]

                masked_input_ids_all[
                    index, selection_index
                ] = self.tokenizer.mask_token_id
                index += 1
                pbar.update(1)
        return masked_input_ids_all

    def generate_dataset(self):
        logger.info("generate dataset from tokenized data")

        if self.arguments.is_dynamic_mask:
            self.training_dataset = EncodingDataset(
                self.training_encodings,
                self.training_span_labels,
                self.training_token_labels,
            )
            self.val_dataset = EncodingDataset(
                self.val_encodings, self.val_span_labels, self.val_token_labels
            )
        else:
            self.training_dataset = MaskedEncodingDataset(
                self.train_mask_input_ids,
                self.training_encodings,
                self.training_span_labels,
                self.training_token_labels,
            )
            self.val_dataset = MaskedEncodingDataset(
                self.val_mask_input_ids,
                self.val_encodings,
                self.val_span_labels,
                self.val_token_labels,
            )

        return self

    def _epoch_train(self, iterator, optim, scheduler=None, is_val=False):
        epoch_loss = 0
        epoch_span_acc = 0
        epoch_token_acc = 0
        if is_val:
            self.model.train(False)
        else:
            self.model.train()

        in_epoch_steps = 0

        with tqdm(total=len(iterator)) as pbar:
            for batch in iterator:
                in_epoch_steps += 1
                pbar.set_description(f"Processing batch: {in_epoch_steps}")

                optim.zero_grad()
                if self.arguments.is_dynamic_mask:
                    input_ids = self.dynamic_mask(batch["input_ids"]).to(self.device)
                else:
                    input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                span_labels = batch["span_labels"].to(self.device)

                if self.arguments.span_only:
                    model_outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        span_labels=span_labels,
                        return_dict=True,
                    )
                    span_prediction_logits = model_outputs.span_prediction_logits
                    loss = model_outputs.loss
                else:
                    labels = batch["labels"].to(self.device)
                    model_outputs = self.model(
                        input_ids,
                        attention_mask=attention_mask,
                        span_labels=span_labels,
                        labels=labels,
                        return_dict=True,
                    )
                    prediction_logits = model_outputs.prediction_logits
                    span_prediction_logits = model_outputs.span_prediction_logits
                    loss = model_outputs.loss

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

                if self.arguments.span_only:
                    epoch_span_acc += self._accuracy(
                        span_prediction_logits, span_labels
                    )
                else:
                    epoch_span_acc += self._accuracy(
                        span_prediction_logits, span_labels
                    )
                    epoch_token_acc += self._accuracy(prediction_logits, labels)

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Last_loss": f"{loss:.2f}",
                        "Avg_cum_loss": f"{epoch_loss/in_epoch_steps:.2f}",
                    }
                )

        return {
            "epoch_loss": epoch_loss / in_epoch_steps,
            "epoch_span_acc": epoch_span_acc / in_epoch_steps,
            "epoch_token_acc": 0
            if self.arguments.span_only
            else epoch_token_acc / in_epoch_steps,
        }

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

    def train(self):
        logger.info("start training")

        train_loader = DataLoader(
            self.training_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        optim = AdamW(self.model.parameters(), lr=1e-5)

        scheduler = get_constant_schedule_with_warmup(
            optim,
            num_warmup_steps=self.arguments.warm_up_steps,
        )

        best_valid_loss = float('inf')
        no_improvement_count = 0
        self.total_steps = 0

        with tqdm(total=self.arguments.epoch) as pbar:
            for epoch in range(self.arguments.epoch):
                pbar.set_description(f"Processing epoch: {epoch + 1}")

                start_time = time.time()

                train_epoch_outputs = self._epoch_train(train_loader, optim, scheduler)
                val_epoch_outputs = self._epoch_train(
                    val_loader, optim, scheduler, True
                )

                self.tensorboard_writter.add_scalar(
                    "Epoch Loss/train", train_epoch_outputs["epoch_loss"], epoch + 1
                )
                self.tensorboard_writter.add_scalar(
                    "Epoch Loss/valid", val_epoch_outputs["epoch_loss"], epoch + 1
                )

                self.tensorboard_writter.add_scalar(
                    "Epoch span acc/train",
                    train_epoch_outputs["epoch_span_acc"],
                    epoch + 1,
                )
                self.tensorboard_writter.add_scalar(
                    "Epoch span acc/valid",
                    val_epoch_outputs["epoch_span_acc"],
                    epoch + 1,
                )

                if not self.arguments.span_only:
                    self.tensorboard_writter.add_scalar(
                        "Epoch token acc/train",
                        train_epoch_outputs["epoch_token_acc"],
                        epoch + 1,
                    )
                    self.tensorboard_writter.add_scalar(
                        "Epoch token acc/valid",
                        val_epoch_outputs["epoch_token_acc"],
                        epoch + 1,
                    )

                end_time = time.time()

                epoch_mins, epoch_secs = self._epoch_time(start_time, end_time)

                logger.info(
                    f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s"
                )
                logger.info(f"Train Loss: {train_epoch_outputs['epoch_loss']:.3f}")
                logger.info(f"Val. Loss: { val_epoch_outputs['epoch_loss']:.3f}")
                logger.info(
                    f"Train Span Acc: {train_epoch_outputs['epoch_span_acc'] * 100:.2f}%"
                )
                logger.info(
                    f"Val. Span Acc: {val_epoch_outputs['epoch_span_acc'] * 100:.2f}%"
                )
                if not self.arguments.span_only:
                    logger.info(
                        f"Train Token Acc: {train_epoch_outputs['epoch_token_acc'] * 100:.2f}%"
                    )
                    logger.info(
                        f"Val. Token Acc: {val_epoch_outputs['epoch_token_acc'] * 100:.2f}%"
                    )

                pbar.update(1)

                if (epoch + 1) % self.arguments.intermediate_persist_step == 0:
                    logger.info(
                        f"Save the intermediate checkpoint for epoch: {epoch + 1}"
                    )
                    self._intermediate_persist(epoch + 1)

                if val_epoch_outputs["epoch_loss"] < best_valid_loss:
                    best_valid_loss = val_epoch_outputs["epoch_loss"]
                    if self.is_parallel:
                        self.best_state_dict = self.model.module.state_dict()
                    else:
                        self.best_state_dict = self.model.state_dict()
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if (
                        self.arguments.early_stop_count > 0
                        and no_improvement_count >= self.arguments.early_stop_count
                    ):
                        logger.info(
                            f"No improvement for past {no_improvement_count} epochs, early stop training."
                        )
                        return self

            logger.info("fine-tune finished")

        self.tensorboard_writter.flush()

        return self

    def _intermediate_persist(self, epoch_index):
        if self.is_parallel:
            persist_moodel = self.model.module
        else:
            persist_moodel = self.model
        torch.save(
            persist_moodel.state_dict(),
            os.path.join(
                self.arguments.model_storage_dir, f"epoch_{epoch_index}_whole_model.bin"
            ),
        )
        torch.save(
            persist_moodel.lm_model.state_dict(),
            os.path.join(
                self.arguments.model_storage_dir, f"epoch_{epoch_index}_lm_only.bin"
            ),
        )

    def persist(self):
        """Persist this model into the passed directory."""

        logger.info("persist fine-tuned model")

        # self.classifier.load_state_dict(self.best_state_dict)
        # self.classifier.save_pretrained(self.arguments.model_storage_dir)

        output_config_file = (
            f"{self.arguments.model_storage_dir}/pretrained_model_config.json"
        )
        self.plm_model_config.to_json_file(output_config_file, use_diff=True)

        torch.save(
            self.best_state_dict,
            os.path.join(self.arguments.model_storage_dir, "whole_model.bin"),
        )

        if self.is_parallel:
            self.model.module.load_state_dict(self.best_state_dict)
            torch.save(
                self.model.module.lm_model.state_dict(),
                os.path.join(self.arguments.model_storage_dir, "lm_only.bin"),
            )
        else:
            self.model.load_state_dict(self.best_state_dict)
            torch.save(
                self.model.lm_model.state_dict(),
                os.path.join(self.arguments.model_storage_dir, "lm_only.bin"),
            )

        logger.info(f"model stored to {self.arguments.model_storage_dir}")

import json
import logging
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from pydantic import BaseModel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AdamW, get_constant_schedule_with_warmup

from punctuator.utils import Models

from .model import MultiLabelNextSentencePrediction
from .punctuation_data_process import EncodingDataset, read_data

logger = logging.getLogger(__name__)
DEFAULT_LABEL_WEIGHT = 1


class TrainingArguments(BaseModel):
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
        addtional_model_config(Optional[Dict]): additional configuration for model
    """

    # basic args
    training_data_path: str
    validation_data_path: str
    plm_model: Optional[Models] = Models.BERT_TOKEN_CLASSIFICATION
    plm_model_weight_name: str
    tokenizer_name: str

    # data args
    target_sequence_length: int = 200
    lower_boundary: int = 5
    # training ars
    epoch: int
    batch_size: int
    model_storage_dir: str
    label2id: Dict
    early_stop_count: Optional[int] = 3
    use_gpu: Optional[bool] = True
    gpu_device: Optional[int] = os.environ.get("CUDA_VISIBLE_DEVICES", 0)
    warm_up_steps: int = 1000
    r_drop: bool = False
    r_alpha: int = 0
    plot_steps: int = 50
    tensorboard_log_dir: Optional[str] = "runs"

    # model args
    addtional_model_config: Optional[Dict]


class TrainingPipeline:
    def __init__(self, training_arguments, verbose=False):
        """Training pipeline

        Args:
            training_arguments (TrainingArguments): arguments passed to training pipeline
        """
        self.arguments = training_arguments

        self.label2id = training_arguments.label2id
        self.id2label = {id: label for label, id in self.label2id.items()}
        self.tensorboard_writter = SummaryWriter(training_arguments.tensorboard_log_dir)

        plm_model_collection = training_arguments.plm_model.value

        self.plm_model_config = plm_model_collection.config.from_pretrained(
            training_arguments.plm_model_weight_name,
            label2id=self.label2id,
            id2label=self.id2label,
            num_labels=len(self.id2label),
            use_return_dict=True,
            **training_arguments.addtional_model_config,
        )
        if verbose:
            logger.info(f"Full model config: {self.plm_model_config}")
        self.tokenizer = plm_model_collection.tokenizer.from_pretrained(
            training_arguments.tokenizer_name
        )

        plm_model = plm_model_collection.model.from_pretrained(
            training_arguments.plm_model_weight_name,
            config=self.plm_model_config,
        )

        self.model = MultiLabelNextSentencePrediction(
            self.plm_model_config, plm=plm_model, r_drop=self.arguments.r_drop
        )

        if torch.cuda.is_available() and training_arguments.use_gpu:
            if torch.cuda.device_count() > 1:
                self.model = torch.nn.DataParallel(self.model)
                self.model.cuda()
            else:
                self.device = torch.device(f"cuda:{training_arguments.gpu_device}")
                self.model.to(self.device)

        else:
            self.device = torch.device("cpu")

    def tokenize(self):
        logger.info("tokenize data")

        with open(self.arguments.training_data_path, "r") as json_file:
            training_source_data = json.load(json_file)

        with open(self.arguments.validation_data_path, "r") as json_file:
            validation_source_data = json.load(json_file)

        self.training_encodings, self.training_tags = read_data(
            source_data=training_source_data,
            tokenizer=self.tokenizer,
            target_sequence_length=self.arguments.target_sequence_length,
            lower_boundary=self.arguments.lower_boundary,
        )
        self.training_encodings = self._encodings_to_tensor(self.training_encodings)
        self.training_tags = [self.label2id[tag] for tag in self.training_tags]

        self.validation_encodings, self.validation_tags = read_data(
            source_data=validation_source_data,
            tokenizer=self.tokenizer,
            target_sequence_length=self.arguments.target_sequence_length,
            lower_boundary=self.arguments.lower_boundary,
        )
        self.validation_encodings = self._encodings_to_tensor(self.validation_encodings)
        self.validation_tags = [self.label2id[tag] for tag in self.validation_tags]

        return self

    def _encodings_to_tensor(self, encodings):
        pbar = tqdm(encodings)
        tensor_encodings = []
        for encoding in pbar:
            tensor_encodings.append(
                {
                    key: torch.tensor(value) if key != "tokens" else value
                    for key, value in encoding.items()
                }
            )
        return tensor_encodings

    def generate_dataset(self):
        logger.info("generate dataset from tokenized data")
        self.training_dataset = EncodingDataset(
            self.training_encodings, self.training_tags
        )
        self.val_dataset = EncodingDataset(
            self.validation_encodings, self.validation_tags
        )

        return self

    def _epoch_train(self, iterator, optim, scheduler=None, is_val=False):
        epoch_loss = 0
        epoch_acc = 0
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
                input_ids = batch["input_ids"].to(self.device)
                token_type_ids = batch["token_type_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                model_outputs = self.model(
                    input_ids,
                    token_type_ids=token_type_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                logits = model_outputs.logits
                loss = model_outputs.loss

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
                epoch_acc += self._accuracy(logits, labels)

                pbar.update(1)
                pbar.set_postfix(
                    {
                        "Last_loss": f"{loss:.2f}",
                        "Avg_cum_loss": f"{epoch_loss/in_epoch_steps:.2f}",
                    }
                )

        return epoch_loss / in_epoch_steps, epoch_acc / in_epoch_steps

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time // 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _accuracy(self, logits, labels):
        if self.device.type == "cuda":
            max_preds = logits.argmax(dim=-1).detach().cpu().numpy().flatten()
        else:
            max_preds = logits.argmax(dim=-1).detach().numpy().flatten()

        if max_preds.shape[0] == labels.shape[0]:
            return np.sum(max_preds == labels) / max_preds.shape[0]

        return 0

    def train(self):
        logger.info("start training")

        train_loader = DataLoader(
            self.training_dataset, batch_size=self.arguments.batch_size, shuffle=False
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.arguments.batch_size, shuffle=False
        )
        optim = AdamW(self.model.parameters(), lr=1e-5)

        scheduler = get_constant_schedule_with_warmup(
            optim,
            num_warmup_steps=self.arguments.warm_up_steps,
        )

        best_valid_loss = 100
        no_improvement_count = 0
        self.total_steps = 0

        with tqdm(total=self.arguments.epoch) as pbar:
            for epoch in range(self.arguments.epoch):
                pbar.set_description(f"Processing epoch: {epoch + 1}")

                start_time = time.time()

                train_loss, train_acc = self._epoch_train(
                    train_loader, optim, scheduler
                )
                val_loss, val_acc = self._epoch_train(
                    val_loader, optim, scheduler, True
                )

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
                    f"Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s"
                )
                logger.info(
                    f"\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc * 100:.2f}%"
                )
                logger.info(
                    f"\t Val. Loss: {val_loss:.3f} |  Val. Acc: {val_acc * 100:.2f}%"
                )
                pbar.update(1)

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    try:
                        self.best_state_dict = self.model.module.state_dict()
                    except AttributeError:
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

    def persist(self):
        """Persist this model into the passed directory."""

        logger.info("persist fine-tuned model")

        # self.classifier.load_state_dict(self.best_state_dict)
        # self.classifier.save_pretrained(self.arguments.model_storage_dir)

        self.model_config.save_pretrained(self.arguments.model_storage_dir)
        torch.save(
            self.best_state_dict,
            os.path.join(self.arguments.model_storage_dir, "pytorch_model.bin"),
        )

        logger.info(f"fine-tuned model stored to {self.arguments.model_storage_dir}")

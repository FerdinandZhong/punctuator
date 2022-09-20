import logging
import time
from os import environ
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pydantic import BaseModel
from sklearn.utils import class_weight
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import (
    AdamW,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    get_constant_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

DEFAULT_LABEL_WEIGHT = 1


class EncodingDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class NERTrainingArguments(BaseModel):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Args:
        # basic arguments
        training_corpus(List[List[str]]): list of sequences for training, longest sequence should be no longer than pretrained LM # noqa: E501
        validation_corpus(List[List[str]]): list of sequences for validation, longest sequence should be no longer than pretrained LM # noqa: E501
        training_tags(List[List[int]]): tags(int) for training
        validation_tags(List[List[int]]): tags(int) for validation
        model_name_or_path(str): name or path of pre-trained model
        tokenizer_name(str): name of pretrained tokenizer

        # training arguments
        epoch(int): number of epoch
        batch_size(int): batch size
        model_storage_dir(str): fine-tuned model storage path
        label2id(Dict): the tags label and id mapping
        early_stop_count(int): after how many epochs to early stop training if valid loss not become smaller. default 3 # noqa: E501
        gpu_device(int): specific gpu card index, default is the CUDA_VISIBLE_DEVICES from environ
        warm_up_steps(int): warm up steps.
        r_drop(bool): whether to train with r-drop
        r_alpha(int): alpha value for kl divengence in the loss, default is 0
        plot_steps(int): record training status to tensorboard among how many steps
        tensorboard_log_dir(Optional[str]): the tensorboard logs output directory, default is "runs"

        # model arguments
        addtional_model_config(Optional[Dict]): additional configuration for model
    """

    # basic args
    training_corpus: List[List[str]]
    validation_corpus: List[List[str]]
    training_tags: List[List[int]]
    validation_tags: List[List[int]]
    model_name_or_path: str
    tokenizer_name: str

    # training ars
    epoch: int
    batch_size: int
    model_storage_dir: str
    label2id: Dict
    early_stop_count: Optional[int] = 3
    gpu_device: int = environ.get("CUDA_VISIBLE_DEVICES", 0)
    warm_up_steps: int = 1000
    r_drop: bool = False
    r_alpha: int = 0
    plot_steps: int = 50
    tensorboard_log_dir: Optional[str] = "runs"

    # model args
    addtional_model_config: Optional[Dict]


class NERTrainingPipeline:
    def __init__(self, training_arguments):
        """Training pipeline for fine-tuning the distilbert token classifier for punctuation

        Args:
            training_arguments (TrainingArguments): arguments passed to training pipeline
        """
        self.arguments = training_arguments
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{training_arguments.gpu_device}")
        else:
            self.device = torch.device("cpu")
        self.label2id = training_arguments.label2id
        self.id2label = {id: label for label, id in self.label2id.items()}
        self.tensorboard_writter = SummaryWriter(training_arguments.tensorboard_log_dir)

    def tokenize(self):
        logger.info("tokenize data")
        self.model_config = DistilBertConfig.from_pretrained(
            self.arguments.model_name_or_path,
            label2id=self.label2id,
            id2label=self.id2label,
            num_labels=len(self.id2label),
            **self.arguments.addtional_model_config,
        )
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.arguments.tokenizer_name,
        )
        self.train_encodings = tokenizer(
            self.arguments.training_corpus,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
        )
        self.val_encodings = tokenizer(
            self.arguments.validation_corpus,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding=True,
        )

        all_ner_tag_ids = [
            tag_id
            for sen_tag_ids in self.arguments.training_tags
            + self.arguments.validation_tags
            for tag_id in sen_tag_ids
        ]
        unique_tag_ids = set(all_ner_tag_ids)

        logger.info(f"unique tag ids: {unique_tag_ids}, id2label: {self.id2label}")

        weights = [
            weight if weight > 0 else DEFAULT_LABEL_WEIGHT
            for weight in np.log(
                class_weight.compute_class_weight(
                    "balanced", classes=list(unique_tag_ids), y=all_ner_tag_ids
                )
            )
        ]
        logger.info(
            f"class weights: {[round(weight, 2) for weight in weights]}, id2label: {self.id2label}"
        )

        self.class_weights = torch.tensor(weights, dtype=torch.float)

        self.train_encoded_tags = self._encode_tags(
            self.arguments.training_tags, self.train_encodings
        )
        self.validation_encoded_tags = self._encode_tags(
            self.arguments.validation_tags, self.val_encodings
        )

        return self

    def generate_dataset(self):
        logger.info("generate dataset from tokenized data")
        self.train_encodings.pop("offset_mapping")
        self.val_encodings.pop("offset_mapping")
        self.training_dataset = EncodingDataset(
            self.train_encodings, self.train_encoded_tags
        )
        self.val_dataset = EncodingDataset(
            self.val_encodings, self.validation_encoded_tags
        )

        return self

    def fine_tune(self):
        logger.info("start fine tune")
        self.classifier = DistilBertForTokenClassification.from_pretrained(
            self.arguments.model_name_or_path,
            config=self.model_config,
        )
        self.classifier.to(self.device)

        train_loader = DataLoader(
            self.training_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        optim = AdamW(self.classifier.parameters(), lr=1e-5)

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

                self.classifier.train()

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
                    self.best_state_dict = self.classifier.state_dict()
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

        self.classifier.load_state_dict(self.best_state_dict)
        self.classifier.save_pretrained(self.arguments.model_storage_dir)

        logger.info(f"fine-tuned model stored to {self.arguments.model_storage_dir}")

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

    def _compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1),
            F.softmax(p, dim=-1),
            reduction="none",
        )

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        # # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def _train(self, iterator, optim, scheduler=None, is_val=False):
        epoch_loss = 0
        epoch_acc = 0
        if is_val:
            self.classifier.train(False)
        else:
            self.classifier.train()

        in_epoch_steps = 0

        with tqdm(total=len(iterator)) as pbar:
            for batch in iterator:
                in_epoch_steps += 1
                pbar.set_description(f"Processing batch: {in_epoch_steps}")

                optim.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                if self.arguments.r_drop:

                    outputs_1 = self.classifier(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    logits_1 = outputs_1.logits
                    if in_epoch_steps == 1:
                        logger.info(f"logits shape {logits_1.size()}")
                    logits_1 = logits_1.view(-1, logits_1.size(-1))
                    if in_epoch_steps == 1:
                        logger.info(f"viewed logits shape {logits_1.size()}")
                    loss_1 = F.cross_entropy(
                        logits_1,
                        labels.view(-1),
                        weight=self.class_weights.to(self.device),
                        reduction="mean",
                    )

                    outputs_2 = self.classifier(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    logits_2 = outputs_2.logits
                    logits_2 = logits_2.view(-1, logits_2.size(-1))
                    loss_2 = F.cross_entropy(
                        logits_2,
                        labels.view(-1),
                        weight=self.class_weights.to(self.device),
                        reduction="mean",
                    )

                    # cross entropy loss for classifier
                    ce_loss = 0.5 * (loss_1 + loss_2)
                    kl_loss = self._compute_kl_loss(logits_1, logits_2)

                    # carefully choose hyper-parameters
                    loss = ce_loss + self.arguments.r_alpha * kl_loss
                    logits = logits_1.add(logits_2) / 2  # average over two logits

                else:
                    outputs = self.classifier(
                        input_ids, attention_mask=attention_mask, labels=labels
                    )
                    logits = outputs.logits
                    logits = logits.view(-1, logits.size(-1))
                    loss = F.cross_entropy(
                        logits,
                        labels.view(-1),
                        # weight=self.class_weights.to(self.device),
                        reduction="mean",
                    )

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
                epoch_acc += self._accuracy(logits, attention_mask, labels)

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

    def _accuracy(self, logits, attention_mask, labels):
        if self.device.type == "cuda":
            max_preds = logits.argmax(dim=-1).detach().cpu().numpy().flatten()
            flattened_labels = labels.detach().cpu().numpy().flatten()
            flattened_attention = attention_mask.detach().cpu().numpy().flatten()
        else:
            max_preds = logits.argmax(dim=-1).detach().numpy().flatten()
            flattened_labels = labels.detach().numpy().flatten()
            flattened_attention = attention_mask.detach().numpy().flatten()

        not_padding_labels = flattened_labels[flattened_attention == 1]
        not_padding_preds = max_preds[flattened_attention == 1]
        reduce_ignored = not_padding_labels >= 0
        true_labels = not_padding_labels[reduce_ignored]  # remove ignored -100
        true_preds = not_padding_preds[reduce_ignored]

        if true_preds.shape[0] == true_labels.shape[0]:
            return np.sum(true_preds == true_labels) / true_preds.shape[0]

        return 0

    def run(self):
        self.tokenize().generate_dataset().fine_tune().persist()

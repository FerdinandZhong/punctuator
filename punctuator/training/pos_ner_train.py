import logging
import time
import os
import json
from typing import Dict, List, Optional, Union
import math
import numpy as np
import torch
import torch.nn as nn
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
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast,
    get_constant_schedule_with_warmup,
)
from punctuator.utils import ModelCollection, Models
from enum import Enum
from collections import namedtuple
from .general_ner_train import NERTrainingArguments, NERTrainingPipeline
from flair.models import SequenceTagger
from flair.data import Sentence
from .focal_loss import focal_loss

logger = logging.getLogger(__name__)
DEFAULT_LABEL_WEIGHT = 0
X_TAG = 'X'


class EncodingDataset:
    def __init__(self, encodings, pos_taggings, labels):
        self.encodings = encodings
        self.pos_taggings = pos_taggings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["pos_taggings"] = torch.tensor(self.pos_taggings[idx])
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


class FocalLoss(nn.CrossEntropyLoss):
    ''' Focal loss for classification tasks on imbalanced datasets '''

    def __init__(self, gamma, alpha=None, ignore_index=-100, reduction='none'):
        super().__init__(weight=alpha, ignore_index=ignore_index, reduction='none')
        self.reduction = reduction
        self.gamma = gamma

    def forward(self, input_, target):
        cross_entropy = super().forward(input_, target)
        # Temporarily mask out ignore index to '0' for valid gather-indices input.
        # This won't contribute final loss as the cross_entropy contribution
        # for these would be zero.
        target = target * (target != self.ignore_index).long()
        input_prob = torch.gather(F.softmax(input_, 1), 1, target.unsqueeze(1))
        loss = torch.pow(1 - input_prob, self.gamma) * cross_entropy
        return torch.mean(loss) if self.reduction == 'mean' else torch.sum(loss) if self.reduction == 'sum' else loss


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class PosTaggingModel(nn.Module):
    
    def __init__(self, pos_tag_size, embedding_dim, output_tag_size, num_layer=6, num_head=8, dropout=0.3) -> None:
        super().__init__()
        self.model_type = 'Transformer'
        self.embedding = nn.Embedding(
            num_embeddings=pos_tag_size,
            embedding_dim=embedding_dim
        )
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)

        transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_head,
            dropout=dropout
        )

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, num_layer)

        self.output_layer = nn.Linear(embedding_dim, output_tag_size)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.output_layer.bias.data.zero_()
        self.output_layer.weight.data.uniform_(-initrange, initrange)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.output_layer(output)

        return output


class PosNERTrainingArguments(NERTrainingArguments):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Args:
        pos_tagging(str): pos tagging pre-trained model
        pos_tagging_embedding_dim(int): pos tagging embedding dimension
    """

    training_pos_tagging_path: str
    val_pos_tagging_path: str
    pos_tagging: str = "flair/upos-english-fast"
    pos_tagging_embedding_dim: int = 512


class PosNERTrainingPipeline(NERTrainingPipeline):
    def __init__(self, training_arguments):
        """Training pipeline for fine-tuning the distilbert token classifier for punctuation

        Args:
            training_arguments (TrainingArguments): arguments passed to training pipeline
        """
        super().__init__(training_arguments)
        self.pos_tagger = SequenceTagger.load(training_arguments.pos_tagging)
        self.pos_tagging_dictionary = self.pos_tagger.label_dictionary
        tag_size = len(self.pos_tagging_dictionary)
        self.train_pos_tags = []
        self.val_pos_tags = []
        self.pos_tagging_model = PosTaggingModel(
            pos_tag_size=tag_size,
            embedding_dim=training_arguments.pos_tagging_embedding_dim,
            output_tag_size=len(training_arguments.label2id),
            dropout=training_arguments.addtional_model_config["dropout"]
        )

        if torch.cuda.is_available() and training_arguments.use_gpu:
            self.device = torch.device(f"cuda:{training_arguments.gpu_device}")
            self.pos_tagging_model.to(self.device)
        else:
            self.device = torch.device("cpu")

    def _mark_ignored_tokens(self, offset_mapping):
        # create an empty array of -100
        sample_marks = np.ones(len(offset_mapping), dtype=int) * -100
        arr_offset = np.array(offset_mapping)

        # set labels whose first offset position is 0 and the second is not 0, only special tokens second is also 0
        sample_marks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = 0

        return sample_marks

    def generate_pos_tagging(self):
        # train
        if os.path.exists(self.arguments.training_pos_tagging_path):
            with open(self.arguments.training_pos_tagging_path, "r") as jf:
                self.train_pos_tags = json.load(jf)
        else:
            for encoding in tqdm(self.train_encodings[:]):
                sequence = encoding.tokens
                sent = Sentence(sequence)
                self.pos_tagger.predict(sent)
                reduce_ignored_tokens = self._mark_ignored_tokens(encoding.offsets) >= 0
                sent_tags = self.pos_tagging_dictionary.get_idx_for_items([entity.tag if reduce_ignored_tokens[e_index] else X_TAG for e_index, entity in enumerate(sent)])
                # print(tags)
                self.train_pos_tags.append(sent_tags)

            with open(self.arguments.training_pos_tagging_path, "w") as jf:
                json.dump(self.train_pos_tags, jf)

        # validations
        if os.path.exists(self.arguments.val_pos_tagging_path):
            with open(self.arguments.val_pos_tagging_path, "r") as jf:
                self.val_pos_tags = json.load(jf)
        else:
            for encoding in tqdm(self.val_encodings[:]):
                sequence = encoding.tokens
                sent = Sentence(sequence)
                self.pos_tagger.predict(sent)
                reduce_ignored_tokens = self._mark_ignored_tokens(encoding.offsets) >= 0
                sent_tags = self.pos_tagging_dictionary.get_idx_for_items([entity.tag if reduce_ignored_tokens[e_index] else X_TAG for e_index, entity in enumerate(sent)])
                # print(tags)
                self.val_pos_tags.append(sent_tags)

            with open(self.arguments.val_pos_tagging_path, "w") as jf:
                json.dump(self.val_pos_tags, jf)

        return self
        
    def generate_dataset(self):
        logger.info("generate dataset from tokenized data")
        self.train_encodings.pop("offset_mapping")
        self.val_encodings.pop("offset_mapping")
        self.training_dataset = EncodingDataset(
            self.train_encodings, self.train_pos_tags, self.train_encoded_tags
        )
        self.val_dataset = EncodingDataset(
            self.val_encodings, self.val_pos_tags, self.validation_encoded_tags
        )

        return self

    def _pos_train(self, iterator, optim, scheduler=None, is_val=False):
        epoch_loss = 0
        epoch_acc = 0
        if is_val:
            self.pos_tagging_model.train(False)

        in_epoch_steps = 0

        epoch_loss = 0

        with tqdm(total=len(iterator)) as pbar:
            for batch in iterator:
                in_epoch_steps += 1
                pbar.set_description(f"Processing batch: {in_epoch_steps}")

                optim.zero_grad()
                taggings = batch["pos_taggings"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"]

                if self.arguments.r_drop:

                    logits_1 = self.pos_tagging_model(
                        taggings
                    )
                    logits_1 = logits_1.view(-1, logits_1.size(-1))
                    loss_1 = F.cross_entropy(
                        logits_1,
                        labels.view(-1),
                        weight=self.class_weights.to(self.device),
                        reduction="mean",
                    )

                    logits_2 =  self.pos_tagging_model(
                        taggings
                    )
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
                    logits = self.pos_tagging_model(
                        taggings
                    )
                    logits = logits.view(-1, logits.size(-1))
                    # loss = F.cross_entropy(
                    #     logits,
                    #     labels.view(-1),
                    #     weight=self.class_weights.to(self.device),
                    #     reduction="mean",
                    # )
                    loss = self.focal_loss(
                        logits,
                        labels.view(-1)
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

    def pos_training(self):
        logger.info("start pos tagging based training")

        self.focal_loss = focal_loss(
            alpha=self.class_weights,
            gamma=2,
            reduction="mean",
            device=self.device
        )

        train_loader = DataLoader(
            self.training_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        optim = AdamW(self.pos_tagging_model.parameters(), lr=1e-5)

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

                self.pos_tagging_model.train()

                train_loss, train_acc = self._pos_train(train_loader, optim, scheduler)
                val_loss, val_acc = self._pos_train(val_loader, optim, scheduler, True)

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
                        self.best_state_dict = self.pos_tagging_model.module.state_dict()
                    except AttributeError:
                        self.best_state_dict = self.pos_tagging_model.state_dict()
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

    def fine_tune(self):
        logger.info("start fine tune")

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
                    try:
                        self.best_state_dict = self.classifier.module.state_dict()
                    except AttributeError:
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

        # self.classifier.load_state_dict(self.best_state_dict)
        # self.classifier.save_pretrained(self.arguments.model_storage_dir)

        # self.model_config.save_pretrained(self.arguments.model_storage_dir)
        os.makedirs(self.arguments.model_storage_dir, exist_ok=True)
        torch.save(self.best_state_dict, os.path.join(self.arguments.model_storage_dir, "pytorch_model.bin"))

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
                    logger.warning(f"tags: {doc_labels}")
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
        self.tokenize().generate_pos_tagging().generate_dataset().pos_training().persist()

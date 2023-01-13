import logging
from os import environ
from typing import Dict, List, Optional
import os
import json
import numpy as np
import torch
from pydantic import BaseModel
from sklearn.metrics import classification_report
from torch._C import device  # noqa: F401
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast
import torch.nn.functional as F

from punctuator.utils import Models, ModelCollection, NORMAL_TOKEN_TAG
from .evalute import EvaluationArguments, EvaluationPipeline
from .pos_ner_train import PosTaggingModel, EncodingDataset, X_TAG
from flair.models import SequenceTagger
from flair.data import Sentence

logger = logging.getLogger(__name__)


class PosEvaluationArguments(EvaluationArguments):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Args:
        pos_tagging(str): pos tagging pre-trained model
        pos_tagging_embedding_dim(int): pos tagging embedding dimension
    """

    test_pos_tagging_path: str
    pos_tagging: str = "flair/upos-english-fast"
    pos_tagging_embedding_dim: int = 512
     
    # model args
    addtional_model_config: Optional[Dict]

class PosEvaluationPipeline():
    def __init__(self, evaluation_arguments) -> None:
        self.arguments = evaluation_arguments
        self.pos_tagger = SequenceTagger.load(evaluation_arguments.pos_tagging)
        self.pos_tagging_dictionary = self.pos_tagger.label_dictionary
        tag_size = len(self.pos_tagging_dictionary)
        self.train_pos_tags = []
        self.val_pos_tags = []
        self.pos_tagging_model = PosTaggingModel(
            pos_tag_size=tag_size,
            embedding_dim=evaluation_arguments.pos_tagging_embedding_dim,
            output_tag_size=len(evaluation_arguments.label2id),
            dropout=evaluation_arguments.addtional_model_config["dropout"]
        )
        self.pos_tagging_model.load_state_dict(torch.load(f"{evaluation_arguments.model_weight_name}/pytorch_model.bin"))
        model_collection = evaluation_arguments.model.value

        self.tokenizer = model_collection.tokenizer.from_pretrained(
            self.arguments.tokenizer_name
        )
        self.label2id = evaluation_arguments.label2id

        if torch.cuda.is_available() and evaluation_arguments.use_gpu:
            self.device = torch.device(f"cuda:{evaluation_arguments.gpu_device}")
            self.pos_tagging_model.to(self.device)
        else:
            self.device = torch.device("cpu")
        self.test_pos_tags = []

    def _mark_ignored_tokens(self, offset_mapping):
        # create an empty array of -100
        sample_marks = np.ones(len(offset_mapping), dtype=int) * -100
        arr_offset = np.array(offset_mapping)

        # set labels whose first offset position is 0 and the second is not 0, only special tokens second is also 0
        sample_marks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = 0

        return sample_marks

    def generate_pos_tagging(self):
        if os.path.exists(self.arguments.test_pos_tagging_path):
            with open(self.arguments.test_pos_tagging_path, "r") as jf:
                self.test_pos_tags = json.load(jf)
        else:
            for encoding in tqdm(self.encodings[:]):
                sequence = encoding.tokens
                sent = Sentence(sequence)
                self.pos_tagger.predict(sent)
                reduce_ignored_tokens = self._mark_ignored_tokens(encoding.offsets) >= 0
                sent_tags = self.pos_tagging_dictionary.get_idx_for_items([entity.tag if reduce_ignored_tokens[e_index] else X_TAG for e_index, entity in enumerate(sent)])
                # print(tags)
                self.test_pos_tags.append(sent_tags)

            with open(self.arguments.test_pos_tagging_path, "w") as jf:
                json.dump(self.test_pos_tags, jf)
        
        self.dataset = EncodingDataset(self.encodings, self.test_pos_tags, self.evaluation_encoded_tags)

        return self

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

        return self

    def pos_validate(self):
        logger.info("start validation")
        val_loader = DataLoader(
            self.dataset, batch_size=self.arguments.batch_size, shuffle=True
        )
        self.pos_tagging_model.train(False)

        steps = 0
        total_preds = []
        total_labels = []
        with tqdm(total=len(val_loader)) as pbar:
            for batch in val_loader:
                steps += 1
                pbar.set_description(f"Processing batch: {steps}")

                taggings = batch["pos_taggings"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"]

                logits = self.pos_tagging_model(
                    taggings
                )
                logits = logits.view(-1, logits.size(-1))
                print(logits)
                loss = F.cross_entropy(
                    logits,
                    labels.view(-1),
                    # weight=self.class_weights.to(self.device),
                    reduction="mean",
                )

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
        logger.info(f"validation report: \n {report}")

    def run(self):
        self.tokenize().generate_pos_tagging().pos_validate()

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

        return true_preds, true_labels

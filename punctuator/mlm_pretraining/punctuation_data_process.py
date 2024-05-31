import logging
import random
import re
from typing import List, Union

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from punctuator.utils import NORMAL_TOKEN_TAG

SPACE_TOKEN = "[SPACE]"
PUNCT_TOKEN = "[PUNCT]"
SENTENCE_ENDINGS = ["PERIOD", "QUESTION"]
logger = logging.getLogger(__name__)


def _read_data(
    source_data, target_length, is_return_list
) -> Union[List[List], List[str]]:
    def read_line(text_line):
        return text_line.strip().split("\t")

    token_docs = []
    span_label_docs = []

    token_doc = []
    span_label_doc = []
    if isinstance(source_data, List):
        pbar = tqdm(source_data)
    else:
        with open(source_data, "r") as data_file:
            pbar = tqdm(data_file.readlines())
    for index, line in enumerate(pbar):
        processed_line = read_line(line)
        try:
            assert len(processed_line) == 2, "bad line"
            # TODO!: remove the non-english characters (except "'")
            regex = re.compile("[^a-zA-Z0-9-+']")
            token = regex.sub("", processed_line[0])
            if token:
                token_doc.append(token)
            if processed_line[1] == NORMAL_TOKEN_TAG:
                token_doc.append(SPACE_TOKEN)
                span_label_doc.append(0)
            else:
                token_doc.append(PUNCT_TOKEN)
                span_label_doc.append(1)
        except AssertionError:
            logger.warning(f"ignore the bad line: {line}, index: {index}")
            continue
        if len(token_doc) >= target_length:
            if is_return_list:
                token_docs.append(token_doc)
            else:
                token_docs.append("".join(token_doc))
            span_label_docs.append(span_label_doc)
            pbar.update(len(token_doc))
            token_doc = []
            span_label_doc = []
    if is_return_list:
        token_docs.append(token_doc)
    else:
        token_docs.append("".join(token_doc))
    span_label_docs.append(span_label_doc)
    pbar.update(len(token_doc))

    pbar.close()

    return token_docs, span_label_docs


# TODO: update docstring
def process_data(source_data, target_length, is_return_list=True):
    """
    Function for generation of tokenized corpus and relevant tags

    Args:
        source_data(str or List): path of input data or input data
        target_sequence_length(int): target sequence length
    """
    logger.info("load data")
    texts, span_labels = _read_data(
        source_data, target_length, is_return_list=is_return_list
    )
    return texts, span_labels


class MaskedEncodingDataset(Dataset):
    def __init__(self, masked_input_ids, encodings, span_labels, labels=None):
        self.encodings = encodings
        self.masked_encodings = encodings.copy()
        self.masked_encodings["input_ids"] = masked_input_ids
        self.span_labels = span_labels
        self.labels = labels

    def __getitem__(self, idx):
        rand = random.uniform(0, 1)
        # following the BERT's original pretraining method
        if rand >= 0.2:
            item = {
                key: val[idx] if torch.is_tensor(val[idx]) else torch.tensor(val[idx])
                for key, val in self.masked_encodings.items()
            }
        else:
            item = {
                key: val[idx] if torch.is_tensor(val[idx]) else torch.tensor(val[idx])
                for key, val in self.encodings.items()
            }
        item["span_labels"] = torch.tensor(self.span_labels[idx]).type(torch.LongTensor)

        if self.labels is not None:
            item["labels"] = (
                self.labels[idx]
                if torch.is_tensor(self.labels[idx])
                else torch.tensor(self.labels[idx]).type(torch.LongTensor)
            )
        return item

    def __len__(self):
        return len(self.span_labels)


class EncodingDataset(Dataset):
    def __init__(self, encodings, span_labels, labels=None):
        self.encodings = encodings
        self.span_labels = span_labels
        self.labels = labels

    def __getitem__(self, idx):
        rand = random.uniform(0, 1)
        # following the BERT's original pretraining method
        item = {
            key: val[idx] if torch.is_tensor(val[idx]) else torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item["span_labels"] = torch.tensor(self.span_labels[idx]).type(torch.LongTensor)

        if self.labels is not None:
            item["labels"] = (
                self.labels[idx]
                if torch.is_tensor(self.labels[idx])
                else torch.tensor(self.labels[idx]).type(torch.LongTensor)
            )
        return item

    def __len__(self):
        return len(self.span_labels)

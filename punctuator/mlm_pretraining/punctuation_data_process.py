import logging
import random
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from punctuator.utils import NORMAL_TOKEN_TAG

SPACE_TOKEN = "[SPACE]"
PUNCT_TOKEN = "[PUNCT]"
SENTENCE_ENDINGS = ["PERIOD", "QUESTION"]
logger = logging.getLogger(__name__)


def _read_data(source_data, min_target_length, max_target_length) -> List[List]:
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
            token_doc.append(processed_line[0])
            if processed_line[1] == NORMAL_TOKEN_TAG:
                token_doc.append(SPACE_TOKEN)
                span_label_doc.append(0)
            else:
                token_doc.append(PUNCT_TOKEN)
                span_label_doc.append(1)
        except AssertionError:
            logger.warning(f"ignore the bad line: {line}, index: {index}")
            continue
        if (
            len(token_doc) >= min_target_length
            and processed_line[1] in SENTENCE_ENDINGS
        ) or len(token_doc) >= max_target_length:
            token_docs.append(token_doc)
            span_label_docs.append(span_label_doc)
            pbar.update(len(token_doc))
            token_doc = []
            span_label_doc = []
    token_docs.append(token_doc)
    span_label_docs.append(span_label_doc)
    pbar.update(len(token_doc))

    pbar.close()

    return token_docs, span_label_docs


def process_data(source_data, min_target_length, max_target_length):
    """
    Function for generation of tokenized corpus and relevant tags

    Args:
        source_data(str or List): path of input data or input data
        target_sequence_length(int): target sequence length
    """
    logger.info("load data")
    texts, span_labels = _read_data(source_data, min_target_length, max_target_length)
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
            item["labels"] = self.labels[idx] if torch.is_tensor(self.labels[idx]) else torch.tensor(self.labels[idx]).type(torch.LongTensor)
        return item

    def __len__(self):
        return len(self.span_labels)

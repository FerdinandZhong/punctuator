import logging
from random import randint
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from punctuator.utils import NORMAL_TOKEN_TAG

PAD_TOKEN = "[PAD]"
DEFAULT_LABEL_WEIGHT = 0.5
logger = logging.getLogger(__name__)


def _read_data(source_data, min_sequence_length, max_sequence_length) -> List[List]:
    def read_line(text_line):
        return text_line.strip().split("\t")

    token_docs = []
    tag_docs = []
    line_index = 0

    token_doc = []
    tag_doc = []
    if isinstance(source_data, List):
        pbar = tqdm(source_data)
    else:
        with open(source_data, "r") as data_file:
            pbar = tqdm(data_file.readlines())
    for index, line in enumerate(pbar):
        if line_index == 0:
            token_doc = []
            tag_doc = []
            target_sequence_length = randint(min_sequence_length, max_sequence_length)
        if line == "\n":
            token_docs.append(token_doc)
            tag_docs.append(tag_doc)
            line_index = 0
            pbar.update(len(token_doc))
            continue
        processed_line = read_line(line)
        try:
            assert len(processed_line) == 2, "bad line"
            token_doc.append(processed_line[0])
            tag_doc.append(processed_line[1])
        except AssertionError:
            logger.warning(f"ignore the bad line: {line}, index: {index}")
            continue
        line_index += 1
        if line_index == target_sequence_length:
            try:
                _verify_senquence(token_doc, min_sequence_length, max_sequence_length)
                _verify_senquence(token_doc, min_sequence_length, max_sequence_length)
                token_docs.append(token_doc)
                tag_docs.append(tag_doc)
                line_index = 0
            except AssertionError:
                logger.warning(f"error generating sequence: {token_doc}")
                line_index = 0
                continue
            pbar.update(target_sequence_length)
    token_doc += [PAD_TOKEN] * (target_sequence_length - line_index)
    tag_doc += [NORMAL_TOKEN_TAG] * (target_sequence_length - line_index)
    try:
        _verify_senquence(token_doc, min_sequence_length, max_sequence_length)
        _verify_senquence(token_doc, min_sequence_length, max_sequence_length)
        token_docs.append(token_doc)
        tag_docs.append(tag_doc)
    except AssertionError:
        logger.warning(f"error generating sequence: {token_doc}")
    
    pbar.close()

    return token_docs, tag_docs

def _verify_senquence(sequence, min_sequence_length, max_sequence_length):
    assert min_sequence_length <= len(sequence) and len(sequence) <= max_sequence_length, "wrong sequence length"

def generate_punctuator_tag_mappings(tag_docs):
    """
    Function for geneartion of label2id mapping if no mappings is provided

    Args:
        tag_docs(List[List[str]]): list of sequences of tags
    """
    all_tags = [tag for tags in tag_docs for tag in tags]
    unique_tags = np.unique(all_tags)
    label2id = {tag: id for id, tag in enumerate(unique_tags)}

    return label2id


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p].tolist(), b[p].tolist()


def process_data(source_data, min_sequence_length, max_sequence_length):
    """
    Function for generation of tokenized corpus and relevant tags

    Args:
        source_data(str or List): path of input data or input data
        min_sequence_length(int): min sequence length of one sample
        max_sequence_length(int): max sequence length of one sample
    """
    logger.info("load data")
    texts, tags = _read_data(
        source_data,
        min_sequence_length,
        max_sequence_length,
    )
    return texts, tags


def generate_training_data_splitting(
    source_data, min_sequence_length, max_sequence_length, split_rate=None
):
    """
    Function for generation of training assets (with splitting) including
    - training corpus
    - training tags
    - validation corpus
    - validation tags

    Args:
        source_data(str or List): path of input data or input data
        min_sequence_length(int): min sequence length of one sample
        max_sequence_length(int): max sequence length of one sample
        split_rate(float): train and validation split rate
    """
    logger.info("load training data")
    texts, tags = _read_data(
        source_data,
        min_sequence_length,
        max_sequence_length,
    )

    logger.info(f"data sample: {texts[0]}")

    (
        train_texts,
        val_texts,
        train_tags,
        val_tags,
    ) = train_test_split(texts, tags, test_size=split_rate, random_state=7)

    return train_texts, train_tags, val_texts, val_tags


class EncodingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

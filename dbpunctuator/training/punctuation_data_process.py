import logging
from random import randint
from typing import List

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

from dbpunctuator.utils import NORMAL_TOKEN_TAG

PAD_TOKEN = "[PAD]"
DEFAULT_LABEL_WEIGHT = 0.5
logger = logging.getLogger(__name__)


def _read_data(file_path, min_sequence_length, max_sequence_length) -> List[List]:
    def read_line(text_line):
        return text_line.strip().split("\t")

    token_docs = []
    tag_docs = []
    line_index = 0

    token_doc = []
    tag_doc = []
    with open(file_path, "r") as data_file:
        pbar = tqdm(data_file.readlines())
        for line in pbar:
            if line_index == 0:
                token_doc = []
                tag_doc = []
                target_sequence_length = randint(
                    min_sequence_length, max_sequence_length
                )
            if line == "\n":
                token_docs.append(token_doc)
                tag_docs.append(tag_doc)
                line_index = 0
                pbar.update(len(token_doc))
                continue
            processed_line = read_line(line)
            try:
                token_doc.append(processed_line[0])
                tag_doc.append(processed_line[1])
            except IndexError:
                logger.warning(f"ignore the bad line: {line}")
                continue
            line_index += 1
            if line_index == target_sequence_length:
                token_docs.append(token_doc)
                tag_docs.append(tag_doc)
                line_index = 0
                pbar.update(target_sequence_length)
        token_doc += [PAD_TOKEN] * (target_sequence_length - line_index)
        tag_doc += [NORMAL_TOKEN_TAG] * (target_sequence_length - line_index)
        token_docs.append(token_doc)
        tag_docs.append(tag_doc)

        pbar.close()

    return token_docs, tag_docs


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


def generate_evaluation_data(data_file_path, min_sequence_length, max_sequence_length):
    """
    Function for generation of evaluation assets including
    - evaluation corpus
    - evaluatio  tags

    Args:
        data_file_path(str): path of overall input data
        min_sequence_length(int): min sequence length of one sample
        max_sequence_length(int): max sequence length of one sample
    """
    logger.info("load validation data")
    texts, tags = _read_data(
        data_file_path,
        min_sequence_length,
        max_sequence_length,
    )
    return texts, tags


def generate_training_data(
    data_file_path, min_sequence_length, max_sequence_length, split_rate
):
    """
    Function for generation of training assets including
    - training corpus
    - training tags
    - validation corpus
    - validation tags

    Args:
        data_file_path(str): path of overall input data
        min_sequence_length(int): min sequence length of one sample
        max_sequence_length(int): max sequence length of one sample
        split_rate(float): train and validation split rate
    """
    logger.info("load training data")
    texts, tags = _read_data(
        data_file_path,
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

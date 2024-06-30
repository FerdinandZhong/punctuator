import logging
import re
from random import randint
from typing import List, Union

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _read_data(
    source_data, min_sequence_length, max_sequence_length
) -> Union[List[List], List[str]]:
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
        if line == "\n":
            token_docs.append(token_doc)
            tag_docs.append(tag_doc)
            pbar.update(len(token_doc))
            token_doc = []
            tag_doc = []
            continue
        processed_line = read_line(line)
        try:
            assert len(processed_line) == 2, "bad line"
            regex = re.compile("[^\u4e00-\u9fa5a-zA-Z0-9-+']")
            token = regex.sub("", processed_line[0])
            if token:
                token_doc.append(token)
                tag_doc.append(processed_line[1])
        except AssertionError:
            logger.warning(f"ignore the bad line: {line}, index: {index}")
            continue
        line_index += 1
        target_sequence_length = randint(min_sequence_length, max_sequence_length)
        if len(token_doc) >= target_sequence_length:
            try:
                _verify_senquence(token_doc, target_sequence_length)
                _verify_senquence(tag_doc, target_sequence_length)
                token_docs.append(token_doc)
                tag_docs.append(tag_doc)
                token_doc = []
                tag_doc = []
            except AssertionError:
                logger.warning(f"error generating sequence: {token_doc}")
                token_doc = []
                tag_doc = []
                continue
            pbar.update(len(token_doc))
    try:
        assert len(token_doc) == len(tag_doc), "Not equal length"
        token_docs.append(token_doc)
        tag_docs.append(tag_doc)
        pbar.update(len(token_doc))
    except AssertionError:
        logger.warning(f"error generating sequence: {token_doc}")

    pbar.close()

    return token_docs, tag_docs


def _verify_senquence(sequence, target_sequence_length):
    assert target_sequence_length <= len(sequence), "wrong sequence length"


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
    Processes the input data to generate sequences of texts and corresponding tags within the specified minimum and maximum sequence lengths.

    This function reads the input data, processes each line to extract tokens and their corresponding tags, and then filters these sequences to ensure they fall within the specified range of sequence lengths. Sequences that do not meet the criteria are discarded.

    Args:
        source_data (str or List): Path to the input data file or a list of input data lines.
        min_sequence_length (int): Minimum allowed length for a sequence.
        max_sequence_length (int): Maximum allowed length for a sequence.

    Returns:
        tuple: A tuple containing two lists, where the first list contains the processed text sequences and the second list contains the corresponding tag sequences.
    """  # noqa E501
    logger.info("load data")
    texts, tags = _read_data(
        source_data,
        min_sequence_length=min_sequence_length,
        max_sequence_length=max_sequence_length,
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
        min_sequence_length (int): Minimum allowed length for a sequence.
        max_sequence_length (int): Maximum allowed length for a sequence.
        split_rate(float): train and validation split rate
    """
    logger.info("load training data")
    texts, tags = _read_data(
        source_data,
        min_sequence_length=min_sequence_length,
        max_sequence_length=max_sequence_length,
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
        item = {key: torch.Tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = (
            self.labels[idx]
            if torch.is_tensor(self.labels[idx])
            else torch.tensor(self.labels[idx]).type(torch.LongTensor)
        )
        return item

    def __len__(self):
        return len(self.labels)

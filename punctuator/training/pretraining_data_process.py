import logging
import re
from random import randint
from typing import List, Union

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from punctuator.utils import DEFAULT_ENGLISH_NER_MAPPING, NORMAL_TOKEN_TAG

logger = logging.getLogger(__name__)
cls_token = "[CLS]"


def _read_data(
    source_data, min_sequence_length, max_sequence_length
) -> Union[List[List], List[List]]:
    def read_line(text_line):
        return text_line.strip().split("\t")

    punct_mapping = {value: key for key, value in DEFAULT_ENGLISH_NER_MAPPING.items()}
    token_docs = []
    has_punct_list = []
    line_index = 0

    token_doc = [cls_token]
    has_punct = 0
    has_punct_num = {0: 0, 1: 0}

    if isinstance(source_data, List):
        pbar = tqdm(source_data)
    else:
        with open(source_data, "r") as data_file:
            pbar = tqdm(data_file.readlines())
    for index, line in enumerate(pbar):
        if line == "\n":
            continue
        processed_line = read_line(line)
        try:
            assert len(processed_line) == 2, "bad line"
            regex = re.compile("[^\u4e00-\u9fa5a-zA-Z0-9-+']")
            token = regex.sub("", processed_line[0])
            if token:
                token_doc.append(token)
                if processed_line[1] != NORMAL_TOKEN_TAG:
                    token_doc.append(punct_mapping[processed_line[1]])
                    has_punct = 1
        except AssertionError:
            logger.warning("ignore the bad line: %s, index: %d", line, index)
            continue
        line_index += 1
        target_sequence_length = randint(min_sequence_length, max_sequence_length)
        if len(token_doc) >= target_sequence_length:
            try:
                _verify_senquence(token_doc, target_sequence_length)
                token_docs.append(token_doc)
                has_punct_num[has_punct] += 1
                has_punct_list.append([has_punct])
                token_doc = []
                has_punct = 0
            except AssertionError:
                logger.warning("error generating sequence: %s", token_doc)
                token_doc = []
                has_punct = 0
                continue
            pbar.update(len(token_doc))
    try:
        token_docs.append(token_doc)
        has_punct_num[has_punct] += 1
        has_punct_list.append([has_punct])
        pbar.update(len(token_doc))
    except AssertionError:
        logger.warning("error generating sequence: %s", token_doc)

    pbar.close()

    logger.info("total zero punct samples: %d", has_punct_num[0])
    logger.info("total having punct samples: %d", has_punct_num[1])

    return token_docs, has_punct_list


def _verify_senquence(sequence, target_sequence_length):
    assert target_sequence_length <= len(sequence), "wrong sequence length"


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

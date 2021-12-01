import logging
import os

import pandas as pd
from tqdm import tqdm

from dbpunctuator.utils import (
    ALL_PUNCS,
    DEFAULT_ENGLISH_NER_MAPPING,
    DIGIT_MASK,
    NORMAL_TOKEN_TAG,
)

from .data_cleanning import cleaning_validator, dataframe_data_cleaning

logger = logging.getLogger(__name__)


def cleanup_data_from_csv(
    csv_path,
    target_col,
    output_file_path,
    ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
    additional_to_remove=[],
    special_cleaning_funcs=[],
):
    """clean up training data from csv file

    Args:
        csv_path (string): path of training data csv
        target_col (list or string): columns of training data in csv
        output_file_path (string): path of cleaned data
        ner_mapping (dict, optional): NER mapping of punctuation marks. Defaults to utils.constant.DEFAULT_ENGLISH_NER_MAPPING keys  # noqa: E501
        additional_to_remove (list, optional): additional special characters to remove, default []
        special_cleaning_funcs (List[funcs], optional): additional cleaning funcs to apply to csv data, default []
    """
    target_col = target_col if isinstance(target_col, list) else [target_col]
    dataframe = pd.read_csv(csv_path).dropna(subset=target_col)
    kept_punctuations = [ord(p) for p in set(ner_mapping.keys())]
    removed_punctuations = [p for p in ALL_PUNCS if p not in kept_punctuations] + [
        ord(p) for p in additional_to_remove
    ]
    logger.info("clean up original data")
    try:
        os.remove(output_file_path)
    except FileNotFoundError:
        pass
    for col in target_col:
        result_df = dataframe_data_cleaning(
            dataframe,
            col,
            kept_punctuations,
            removed_punctuations,
            *special_cleaning_funcs,
        )
        with open(output_file_path, "a+") as output_file:
            for row in result_df[col].tolist():
                try:
                    if row and cleaning_validator(
                        row, kept_punctuations, removed_punctuations
                    ):
                        if row[-1] not in ner_mapping:
                            output_file.write("%s . \n" % row)
                        else:
                            output_file.write("%s \n" % row)
                except AssertionError as e:
                    logger.warning(str(e))


def process_line(line, ner_mapping):
    text_list = line.split()
    token_list = []
    tag_list = []
    # clean up puncs in the beginning of the text
    latest_word = text_list.pop(0)
    while latest_word in ner_mapping:
        if not text_list:
            break
        latest_word = text_list.pop(0)
    latest_token = NORMAL_TOKEN_TAG
    latest_is_punc = False
    for word in text_list:
        if word in ner_mapping:
            if not latest_is_punc:
                latest_token = ner_mapping[word]
                latest_is_punc = True
                token_list.append(latest_word)
                tag_list.append(latest_token)
            else:
                pass
        else:
            if not latest_is_punc:
                token_list.append(latest_word)
                tag_list.append(latest_token)
            latest_is_punc = False
            if word.isdigit():
                word = DIGIT_MASK
            latest_word = word
            latest_token = NORMAL_TOKEN_TAG
    if not latest_is_punc:
        token_list.append(latest_word)
        tag_list.append(latest_token)
    return token_list, tag_list


def generate_training_data(
    cleaned_data_path,
    training_data_path,
    ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
):
    """generate "token tag" format training data based on cleaned text

    Args:
        cleaned_data_path (string): path of cleaned data
        training_data_path (string): path of generated training data
        plain_token (string, optional): token for plain token, default "O"
        ner_mapping (dict, optional): ner mapping for puncs and labels
    """
    logger.info("generate training data")
    with open(cleaned_data_path, "r") as data_file:
        lines = data_file.readlines()
    with open(training_data_path, "w+") as training_data_file:
        pbar = tqdm(lines)
        for line in pbar:
            tokens, tags = process_line(line, ner_mapping)
            for token, tag in zip(tokens, tags):
                training_data_file.write("%s\t%s\n" % (token, tag))
        pbar.close()

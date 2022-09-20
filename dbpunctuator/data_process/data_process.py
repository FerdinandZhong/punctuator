import logging
import os
from typing import List

import pandas as pd
from tqdm import tqdm

from dbpunctuator.utils import ALL_PUNCS, DEFAULT_ENGLISH_NER_MAPPING, NORMAL_TOKEN_TAG

from .data_cleanning import (
    cleaning_validator,
    dataframe_data_cleaning,
    text_lines_cleaning,
)

logger = logging.getLogger(__name__)


def cleanup_data_from_csv(
    source_data,
    target_col,
    output_file_path,
    ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
    additional_to_remove=[],
    special_cleaning_funcs=[],
):
    """clean up training data from csv file

    Args:
        source_data (string or Pandas.Dataframe): path of training data csv or own defined dataframe
        target_col (list or string): columns of training data in csv
        output_file_path (string): path of cleaned data
        ner_mapping (dict, optional): NER mapping of punctuation marks. Defaults to utils.constant.DEFAULT_ENGLISH_NER_MAPPING keys  # noqa: E501
        additional_to_remove (list, optional): additional special characters to remove, default []
        special_cleaning_funcs (List[funcs], optional): additional cleaning funcs to apply to csv data, default []
    """
    target_col = target_col if isinstance(target_col, list) else [target_col]
    if isinstance(source_data, pd.DataFrame):
        dataframe = source_data
    else:
        dataframe = pd.read_csv(source_data).dropna(subset=target_col)
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
                        output_file.write(row + " \n")
                except AssertionError as e:
                    logger.warning(str(e))


def clean_up_data_from_txt(
    source_data,
    output_file_path,
    ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
    additional_to_remove=[],
    special_cleaning_funcs=[],
):
    """clean up training data from text file

    Args:
        source_data (string or List): path of training data or own defined data
        output_file_path (string): path of cleaned data
        ner_mapping (dict, optional): NER mapping of punctuation marks. Defaults to utils.constant.DEFAULT_ENGLISH_NER_MAPPING keys  # noqa: E501
        additional_to_remove (list, optional): additional special characters to remove, default []
        special_cleaning_funcs (List[funcs], optional): additional cleaning funcs to apply to csv data, default []
    """
    kept_punctuations = [ord(p) for p in set(ner_mapping.keys())]
    removed_punctuations = [p for p in ALL_PUNCS if p not in kept_punctuations] + [
        ord(p) for p in additional_to_remove
    ]
    if isinstance(source_data, List):
        cleaned_up_lines = list(
            text_lines_cleaning(
                source_data,
                kept_punctuations,
                removed_punctuations,
                *special_cleaning_funcs,
            )
        )
    else:
        with open(source_data, "r") as file:
            cleaned_up_lines = list(
                text_lines_cleaning(
                    file.readlines(),
                    kept_punctuations,
                    removed_punctuations,
                    *special_cleaning_funcs,
                )
            )
    try:
        os.remove(output_file_path)
    except FileNotFoundError:
        pass
    with open(output_file_path, "a+") as output_file:
        for line in cleaned_up_lines:
            try:
                if line and cleaning_validator(
                    line, kept_punctuations, removed_punctuations
                ):
                    output_file.write(line + "\n")
            except AssertionError as e:
                logger.warning(str(e))


def process_line(line, ner_mapping):
    text_list = line.split()
    token_list = []
    tag_list = []
    if len(text_list) == 0:
        return token_list, tag_list
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
            latest_word = word
            latest_token = NORMAL_TOKEN_TAG
    if not latest_is_punc:
        token_list.append(latest_word)
        tag_list.append(latest_token)
    return token_list, tag_list


def generate_corpus(
    cleaned_data_path,
    training_data_path,
    ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
):
    """generate "token tag" format training data based on cleaned text

    Args:
        cleaned_data_path (string): path of cleaned data
        training_data_path (string): path of generated training data
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
            training_data_file.write("\n")
        pbar.close()

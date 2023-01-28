import itertools
import json
import logging
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

PAD_TOKEN = "[PAD]"
CLS_TOKEN = "[CLS]"
SEP_TOKEN = "[SEP]"
SENTENCE_SPLITTINGS = ["COMMA", "PERIOD", "QUESTION"]
logger = logging.getLogger(__name__)


def optimize_lines(input_lines, tokenizer, target_path, verbose=False):
    def read_line(text_line):
        return text_line.strip().split("\t")

    tokenized_data = []
    pbar = tqdm(input_lines)
    write_line = True
    for index, line in enumerate(pbar):
        processed_line = read_line(line)
        try:
            assert len(processed_line) == 2, "bad line"
            if index > 0 and write_line:
                tokenized_data.append([tokenizer.tokenize(last_token), last_tag])
            last_token = processed_line[0]
            last_tag = processed_line[1]
            write_line = True
        except AssertionError:
            if verbose:
                logger.warn(
                    f"find the bad line: {line}, last token: {last_token}, last tag: {last_tag}, index: {index}"
                )
            tokenized_data.append([tokenizer.tokenize(last_token), processed_line[0]])
            write_line = False
            continue
    if write_line:
        tokenized_data.append([tokenizer.tokenize(last_token), last_tag])
    with open(target_path, "w") as output_file:
        json.dump(tokenized_data, output_file, indent=4)


def generate_encoding_pairs(
    tokenizer, tokens, tags, target_sequence_length, lower_boundary
):
    """Function for generating encoding pairs for the chunk of tokens.

    Args:
        tokenizer (_type_): _description_
        tokens (_type_): _description_
        tags (_type_): _description_
        target_sequence_length (_type_): _description_
        lower_boundary (_type_): _description_

    Returns:
        _type_: _description_
    """
    sequence_pairs = []
    paired_tags = []
    starting_token = 0
    total_length = len(tags)
    for inner_line_index, tag in enumerate(tags):
        sequence_1 = tokens[starting_token : inner_line_index + 1]
        flatterned_sequence_1 = (
            [CLS_TOKEN] + list(itertools.chain(*sequence_1)) + [SEP_TOKEN]
        )
        sequence_2 = tokens[inner_line_index + 1 :]
        flatterned_sequence_2 = list(itertools.chain(*sequence_2)) + [SEP_TOKEN]
        total_length = len(flatterned_sequence_1) + len(flatterned_sequence_2)
        padding_length = target_sequence_length - total_length
        input_ids = (
            tokenizer.convert_tokens_to_ids(
                flatterned_sequence_1 + flatterned_sequence_2
            )
            + [0] * padding_length
        )
        token_type_ids = [0] * len(flatterned_sequence_1) + [1] * (
            len(flatterned_sequence_2) + padding_length
        )
        attention_mask = [1] * total_length + [0] * padding_length
        encodings = {
            "tokens": flatterned_sequence_1
            + flatterned_sequence_2
            + [PAD_TOKEN] * padding_length,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
            "attention_mask": attention_mask,
        }
        sequence_pairs.append(encodings)
        paired_tags.append(tag)

        if tag in SENTENCE_SPLITTINGS:
            starting_token = inner_line_index + 1
            if len(tags[starting_token:]) <= lower_boundary:
                return (
                    sequence_pairs,
                    paired_tags,
                    tokens[starting_token:],
                    tags[starting_token:],
                )

    return sequence_pairs, paired_tags, [], []


def read_data(
    source_data, tokenizer, target_sequence_length, lower_boundary, verbose=False
) -> List[List]:
    """Function for generation of tokenized corpus and relevant tags
    Each corpus will be in the format of
    {
        "tokens": ,
        "input_ids": ,
        "token_type_ids": ,
        "attention_mask"; ,
    }

    Args:
        source_data (_type_): _description_
        tokenizer (_type_): _description_
        target_sequence_length (_type_): _description_
        lower_boundary (_type_): _description_

    Raises:
        assertion_error: _description_
        e: _description_
        assertion_error: _description_
        e: _description_

    Returns:
        List[List]: _description_
    """
    all_sequence_pairs = []
    all_tags_list = []
    line_index = 0
    next_token_length = len(source_data[0][0])

    pbar = tqdm(total=len(source_data))
    current_sequence_length = 0
    current_tokens = []
    current_tags = []
    while line_index < len(source_data):
        while current_sequence_length + next_token_length < (
            target_sequence_length - 3
        ):
            current_sequence_length += next_token_length
            current_tokens.append(source_data[line_index][0])
            current_tags.append(source_data[line_index][1])
            line_index += 1
            if line_index >= len(source_data):
                break
            next_token_length = len(source_data[line_index][0])
        try:
            (
                sequence_pairs,
                paired_tags,
                current_tokens,
                current_tags,
            ) = generate_encoding_pairs(
                tokenizer,
                current_tokens,
                current_tags,
                target_sequence_length,
                lower_boundary,
            )
            if current_tokens and current_tags:
                if verbose:
                    logger.info(
                        f"Find short remaining partial sentence with index: {line_index}, remaining_tokens: {current_tokens}"
                    )
            current_sequence_length = len(list(itertools.chain(*current_tokens)))
            _verify_senquence_pairs(sequence_pairs, paired_tags, target_sequence_length)
            all_sequence_pairs.extend(sequence_pairs)
            all_tags_list.extend(paired_tags)
        except AssertionError as assertion_error:
            logger.error(
                f"error generating sequence with reason: {str(assertion_error)}"
            )
            raise assertion_error
        except Exception as e:
            logger.error(f"error generating sequence with reason: {str(e)}")
            raise e

        pbar.update(len(paired_tags))

    if current_tokens and current_tags:
        try:
            (
                sequence_pairs,
                paired_tags,
                current_tokens,
                current_tags,
            ) = generate_encoding_pairs(
                tokenizer,
                current_tokens,
                current_tags,
                target_sequence_length,
                lower_boundary,
            )
            current_sequence_length = len(list(itertools.chain(*current_tokens)))
            _verify_senquence_pairs(sequence_pairs, paired_tags, target_sequence_length)
            all_sequence_pairs.extend(sequence_pairs)
            all_tags_list.extend(paired_tags)
        except AssertionError as assertion_error:
            logger.warning(
                f"error generating sequence with reason: {str(assertion_error)}"
            )
            raise assertion_error
        except Exception as e:
            logger.warning(f"error generating sequence with reason: {str(e)}")
            raise e

    pbar.update(len(paired_tags))
    pbar.close()

    return all_sequence_pairs, all_tags_list


def _verify_senquence_pairs(sequence_pairs, tags, target_sequence_length):
    assert (
        len(sequence_pairs) == len(tags) <= target_sequence_length
    ), f"wrong sequence pair length {len(sequence_pairs)}, {len(tags)}"


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p].tolist(), b[p].tolist()


class EncodingDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = self.encodings[idx]
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

import logging
import os
import re
from itertools import zip_longest
from random import randint
from typing import List, Union

from plane import CJK, replace
from plane.pattern import EMAIL
from tqdm import tqdm

from punctuator.data_process.data_cleanning import cleaning_validator
from punctuator.utils import ALL_PUNCS, NORMAL_TOKEN_TAG, URL, clean_digits

EMAIL_TOKEN = "email"
URL_TOKEN = "url"


logger = logging.getLogger(__name__)


def chinese_combine(input, extra_recognized=[",", ".", "!", "?"]):
    """Combine Chinese characters by:
    - Removing space between every Chinese character. Note: English word will remain as original

    Args:
        input (string): text to apply the regex func to
    """

    regex = re.compile("(?P<%s>%s)" % (CJK.name, CJK.pattern), CJK.flag)
    result = ""
    start = 0
    try:
        for t in regex.finditer(input):
            non_cjk = input[start : t.start()].strip()
            length_non_cjk = len(non_cjk)
            if length_non_cjk > 0:
                if length_non_cjk == 1 and non_cjk in extra_recognized:
                    result += non_cjk
                else:
                    result += " " if start > 0 else ""
                    result += non_cjk
                    result += " "

            result += "".join(
                [char for char in list(input[t.start() : t.end()]) if char != " "]
            )
            start = t.end()
        result += " " + input[start:].strip()
    except TypeError as err:
        logger.warning(f"parsing data: {input} with error: {str(err)}")
    return result.strip()


def read_line(text_line):
    return text_line.strip().split("\t")


def sync_lines(groundtruth, bert_outputs):
    groundtruth = [line for line in groundtruth.readlines() if line.strip() != ""]
    new_bert_outputs = []
    for line, bert_line in zip_longest(groundtruth, bert_outputs):
        if read_line(bert_line)[0] == read_line(line)[0]:
            new_bert_outputs.append(read_line(bert_line))
        else:
            new_bert_outputs.append([read_line(line)[0], "O"])


def text_lines_cleaning(
    input_lines, target_punctuations, removed_punctuations, *special_cleaning_funcs
):
    logger.info("clean up text file line by line.")
    logger.info("replace email with email")
    logger.info("replace url with url")

    pbar = tqdm(input_lines)
    for line in pbar:
        if special_cleaning_funcs:
            for func in special_cleaning_funcs:
                try:
                    line = func(line)
                except Exception as err:
                    logger.warning(f"error {str(err)} with func {func} for line {line}")

        line = clean_digits(line)

        line = replace(line, EMAIL, EMAIL_TOKEN)

        line = replace(line, URL, URL_TOKEN)

        translator = str.maketrans({key: None for key in removed_punctuations})
        space_translator = str.maketrans(
            {key: " {0} ".format(chr(key)) for key in target_punctuations}
        )

        yield line.translate(translator).translate(space_translator).strip()

    pbar.close()


def clean_up_data_from_txt(
    source_data,
    output_file_path,
    target_punctuations,
    additional_to_keep=[],
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
    target_punctuations = [ord(p) for p in set(target_punctuations)]
    kept_punctuations = [ord(p) for p in additional_to_keep]
    removed_punctuations = [
        p for p in ALL_PUNCS if p not in (target_punctuations + kept_punctuations)
    ] + [ord(p) for p in additional_to_remove]
    if isinstance(source_data, List):
        cleaned_up_lines = list(
            text_lines_cleaning(
                source_data,
                target_punctuations,
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

    return cleaned_up_lines


def read_data_to_w_special_token(
    source_data,
    bert_output,
    min_sequence_length,
    max_sequence_length,
    punct_special_token,
    is_split_into_words: bool = True,
) -> Union[List[List], List[List]]:
    texts_wo_puncts = []
    texts_labels = []
    texts_w_bert_output = []
    line_index = 0

    text_wo_puncts = []
    text_labelss = []
    text_w_bert_output = []
    with open(source_data, "r", encoding="utf-8") as data_file:
        pbar = tqdm([line for line in data_file.readlines() if line.strip() != ""])
    with open(bert_output, "r", encoding="utf-8") as bert_output_file:
        bert_lines = bert_output_file.readlines()

    assert len(pbar) == len(bert_lines), "total length not matched"
    for index, line in enumerate(pbar):
        if line == "\n":
            continue
        processed_line = read_line(line)
        bert_output_line = bert_lines[index]
        processed_bert_line = read_line(bert_output_line)
        try:
            assert len(processed_line) == 2, "bad line"
            assert len(processed_bert_line) == 2, "bad bert output line"
            # regex = re.compile("[^\u4e00-\u9fa5a-zA-Z0-9-+']")
            token = processed_line[0].lower()
            # if processed_bert_line[0] != token:
            #     print(f"index: {index}, {processed_bert_line}, {token}")
            #     break

            tag = processed_line[1]
            text_labelss.append(tag)
            text_wo_puncts.append(token)
            if processed_bert_line[1].upper() != NORMAL_TOKEN_TAG:
                bert_token = token + punct_special_token
            else:
                bert_token = token
            text_w_bert_output.append(bert_token)

        except AssertionError:
            print(f"ignore the bad line: {line}, index: {index}")
            continue
        line_index += 1
        target_sequence_length = randint(min_sequence_length, max_sequence_length)
        if len(text_wo_puncts) >= target_sequence_length:
            if is_split_into_words:
                texts_wo_puncts.append(text_wo_puncts)
            else:
                texts_wo_puncts.append(
                    chinese_combine(" ".join(text_wo_puncts), [punct_special_token])
                )
            texts_w_bert_output.append(
                chinese_combine(" ".join(text_w_bert_output), [punct_special_token])
            )
            texts_labels.append(text_labelss)

            text_wo_puncts = []
            text_w_bert_output = []
            text_labelss = []
            pbar.update(len(text_wo_puncts))

    try:
        if len(text_wo_puncts) > 0:
            print(text_w_bert_output)
            if is_split_into_words:
                texts_wo_puncts.append(text_wo_puncts)
            else:
                texts_wo_puncts.append(
                    chinese_combine(" ".join(text_wo_puncts), [punct_special_token])
                )
            texts_w_bert_output.append(
                chinese_combine(" ".join(text_w_bert_output), [punct_special_token])
            )
            texts_labels.append(text_labelss)
            pbar.update(len(text_wo_puncts))
    except AssertionError:
        print(f"error generating sequence: {text_wo_puncts}")

    pbar.close()

    return texts_wo_puncts, texts_w_bert_output, texts_labels


from copy import deepcopy


def generate_dataset(chat_messages, input_list, token=None):
    all_samples = []
    if token is not None:
        for message_index, message in enumerate(chat_messages):
            chat_messages[message_index]["content"] = message["content"].format(
                token=token
            )
    for input in input_list:
        current_loop_chat_messages = deepcopy(chat_messages)
        current_loop_chat_messages.append({"role": "user", "content": input})
        all_samples.append(current_loop_chat_messages)

    return all_samples


def normalize_puncs(input, normalization):
    normalizer = re.compile(
        "({})".format("|".join(map(re.escape, normalization.keys())))
    )
    return normalizer.sub(lambda m: normalization[m.string[m.start() : m.end()]], input)


def process_line(line, ner_mapping):
    text_list = line.split()
    word_list = []
    tag_list = []
    if len(text_list) == 0:
        return word_list, tag_list
    # clean up puncs in the beginning of the text
    latest_word = text_list.pop(0)
    while latest_word in ner_mapping:
        if not text_list:
            break
        latest_word = text_list.pop(0)
    latest_tag = NORMAL_TOKEN_TAG
    latest_is_punc = False
    for word in text_list:
        if word in ner_mapping:
            if not latest_is_punc:
                latest_tag = ner_mapping[word]
                latest_is_punc = True
                word_list.append(latest_word.lower())
                tag_list.append(latest_tag)
            else:
                pass
        else:
            if not latest_is_punc:
                word_list.append(latest_word.lower())
                tag_list.append(latest_tag)
            latest_is_punc = False
            latest_word = word
            latest_tag = NORMAL_TOKEN_TAG
    if not latest_is_punc:
        word_list.append(latest_word.lower())
        tag_list.append(latest_tag)
    return word_list, tag_list


def is_sublist_and_get_indices(list1, list2):
    len_list1 = len(list1)
    len_list2 = len(list2)

    if len_list1 == len_list2:
        return True, list(range(len_list1))

    # Iterate over list2 to find the start of list1
    for i in range(len_list2 - len_list1 + 1):
        if (
            list2[i : i + len_list1] == list1
            or " ".join(list2[i : i + len_list1]).lower() == " ".join(list1).lower()
        ):
            # If list1 is found, return True and the indices
            return True, list(range(i, i + len_list1))

    # If not found, return False and an empty list
    return False, []

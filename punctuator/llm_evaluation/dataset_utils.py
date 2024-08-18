import logging
import re
from itertools import zip_longest
from typing import List

from tqdm import tqdm

from punctuator.utils import NORMAL_TOKEN_TAG

logger = logging.getLogger(__name__)


EMAIL_TOKEN = "email"
URL_TOKEN = "url"

from plane import CJK


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


from random import randint
from typing import Union


def read_data_to_w_special_token(
    source_data,
    bert_output,
    min_sequence_length,
    max_sequence_length,
    punct_special_token,
) -> Union[List[List], List[List]]:
    texts_wo_puncts = []
    texts_tokens = []
    texts_w_bert_output = []
    line_index = 0

    text_wo_puncts = []
    text_tokens = []
    text_w_bert_output = []
    with open(source_data, "r", encoding="utf-8") as data_file:
        pbar = tqdm([line for line in data_file.readlines() if line.strip() != ""])
    with open(bert_output, "r", encoding="utf-8") as bert_output_file:
        bert_lines = bert_output_file.readlines()

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
            text_tokens.append(tag)
            text_wo_puncts.append(token)
            if processed_bert_line[1].upper() != NORMAL_TOKEN_TAG:
                bert_token = token + punct_special_token
            else:
                bert_token = token
            text_w_bert_output.append(bert_token.lower())

        except AssertionError:
            print(f"ignore the bad line: {line}, index: {index}")
            continue
        line_index += 1
        target_sequence_length = randint(min_sequence_length, max_sequence_length)
        if len(text_wo_puncts) >= target_sequence_length:
            texts_wo_puncts.append(
                chinese_combine(" ".join(text_wo_puncts), [punct_special_token])
            )
            texts_w_bert_output.append(
                chinese_combine(" ".join(text_w_bert_output), [punct_special_token])
            )
            texts_tokens.append(text_tokens)

            text_wo_puncts = []
            text_w_bert_output = []
            text_tokens = []
            pbar.update(len(text_wo_puncts))

    try:
        if len(text_wo_puncts) > 0:
            print(text_w_bert_output)
            texts_wo_puncts.append(
                chinese_combine(" ".join(text_wo_puncts), [punct_special_token])
            )
            texts_w_bert_output.append(
                chinese_combine(" ".join(text_w_bert_output), [punct_special_token])
            )
            texts_tokens.append(text_tokens)
            pbar.update(len(text_wo_puncts))
    except AssertionError:
        print(f"error generating sequence: {text_wo_puncts}")

    pbar.close()

    return texts_wo_puncts, texts_w_bert_output, texts_tokens


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
                word_list.append(latest_word)
                tag_list.append(latest_tag)
            else:
                pass
        else:
            if not latest_is_punc:
                word_list.append(latest_word)
                tag_list.append(latest_tag)
            latest_is_punc = False
            latest_word = word
            latest_tag = NORMAL_TOKEN_TAG
    if not latest_is_punc:
        word_list.append(latest_word)
        tag_list.append(latest_tag)
    return word_list, tag_list

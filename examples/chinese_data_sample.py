import re
from itertools import zip_longest

from plane import CJK

from dbpunctuator.data_process import (
    clean_up_data_from_txt,
    generate_training_data,
    remove_brackets_text,
)
from dbpunctuator.utils import DEFAULT_CHINESE_NER_MAPPING


# self defined special cleaning func
# as the ch training data used is having en puncs
def normalize_puncs(input):
    normalization = {"?": "？", "!": "！", "（": "(", "）": ")", "...": "。", ",": "，"}
    normalizer = re.compile(
        "({})".format("|".join(map(re.escape, normalization.keys())))
    )
    return normalizer.sub(lambda m: normalization[m.string[m.start() : m.end()]], input)


def remove_title(input):
    """remove title inside training data. (title doesn't have period at the end)"""
    if input.strip() and input.strip()[-1] not in ["。", "？", "！"]:
        return ""
    return input


def revert_ascii_chars_whitespace(input):
    """revert the original data to remove spaces between latin chars

    Args:
        input (string): input to be processed

    """
    regex = re.compile("(?P<%s>%s)" % (CJK.name, CJK.pattern), CJK.flag)
    result = ""
    start = 0
    for t in regex.finditer(input):
        result += " " + "".join(
            [char for char in list(input[start : t.start()]) if char != " "]
        )
        result += " " + input[t.start() : t.end()]
        start = t.end()
    result += input[start:]
    return result


def merge_data(whole_data_path, *tokens_data_paths):
    all_lines = []
    with open(whole_data_path, "w+") as whole_data_file:
        for cleaned_data_path in tokens_data_paths:
            with open(cleaned_data_path, "r") as data_file:
                all_lines.append(data_file.readlines())
        for lines in zip_longest(*all_lines):
            for line in lines:
                if line:
                    whole_data_file.write(line)


if __name__ == "__main__":
    # cleaned Chinese training data
    with open("./original_data/source_BIO_2014_cropus.txt", "r") as file:
        source_data = [
            line[1] for line in enumerate(file.readlines()) if line[0] % 2 == 0
        ]  # only use even index lines, rest for validation
        # need to use lines located at the end of file as they are from differnet news topic
    clean_up_data_from_txt(
        source_data,
        "./training_data/chinese_cleaned_text.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
        additional_to_remove=["℃", "|"],
        special_cleaning_funcs=[
            normalize_puncs,
            revert_ascii_chars_whitespace,
            remove_brackets_text,
            remove_title,
        ],
    )

    generate_training_data(
        "./training_data/chinese_cleaned_text.txt",
        "./training_data/chinese_token_tag_data.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
    )

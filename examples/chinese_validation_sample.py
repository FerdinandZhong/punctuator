import re
import unicodedata

from plane import CJK

from dbpunctuator.data_process import (
    chinese_split,
    clean_up_data_from_txt,
    generate_training_data,
    remove_brackets_text,
)
from dbpunctuator.training import ValidationArguments, ValidationPipeline
from dbpunctuator.utils import DEFAULT_CHINESE_NER_MAPPING


def remove_special_chars(input):
    return re.sub(r"\/[a-z]+", "", input)


def normalize(input):
    return unicodedata.normalize("NFKC", input)


def normalize_puncs(input):
    normalization = {"?": "? ", "!": "！", "（": "(", "）": ")", "...": "。", ",": "，"}
    normalizer = re.compile(
        "({})".format("|".join(map(re.escape, normalization.keys())))
    )
    return normalizer.sub(lambda m: normalization[m.string[m.start() : m.end()]], input)


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


if __name__ == "__main__":
    # test with left MSRA News data
    clean_up_data_from_txt(
        "./original_data/MSRA.txt",
        "./validation_data/chinese_cleaned_text.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
        additional_to_remove=["℃", "|"],
        special_cleaning_funcs=[
            remove_special_chars,
            normalize,
            normalize_puncs,
            revert_ascii_chars_whitespace,
            chinese_split,
            remove_brackets_text,
        ],
    )

    generate_training_data(
        "./validation_data/chinese_cleaned_text.txt",
        "./validation_data/chinese_token_tag_data.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
    )

    validation_args = ValidationArguments(
        data_file_path="validation_data/chinese_token_tag_data.txt",
        model_name_or_path="Qishuai/distilbert_punctuator_zh",
        tokenizer_name="Qishuai/distilbert_punctuator_zh",
        min_sequence_length=100,
        max_sequence_length=200,
        batch_size=16,
    )

    validate_pipeline = ValidationPipeline(validation_args)
    validate_pipeline.run()

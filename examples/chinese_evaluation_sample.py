import re
import unicodedata

from plane import CJK

from dbpunctuator.data_process import clean_up_data_from_txt, generate_corpus
from dbpunctuator.training import (
    EvaluationArguments,
    EvaluationPipeline,
    generate_evaluation_data,
)
from dbpunctuator.utils import (
    DEFAULT_CHINESE_NER_MAPPING,
    chinese_split,
    remove_brackets_text,
)


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
        "./evaluation_data/chinese_cleaned_text.txt",
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

    generate_corpus(
        "./evaluation_data/chinese_cleaned_text.txt",
        "./evaluation_data/chinese_token_tag_data.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
    )

    label2id = (
        {
            "C_COMMA": 2,
            "C_DUNHAO": 1,
            "C_EXLAMATIONMARK": 0,
            "C_PERIOD": 5,
            "C_QUESTIONMARK": 3,
            "O": 4,
        },
    )
    evalution_corpus, evaluation_tags = generate_evaluation_data(
        "evaluation_data/chinese_token_tag_data.txt", 16, 256
    )
    evaluation_tags = [[label2id[tag] for tag in doc] for doc in evaluation_tags]
    evaluation_args = EvaluationArguments(
        evaluation_corpus=evalution_corpus,
        evaluation_tags=evaluation_tags,
        model_name_or_path="Qishuai/distilbert_punctuator_zh",
        tokenizer_name="Qishuai/distilbert_punctuator_zh",
        batch_size=16,
        gpu_device=2,
        label2id=label2id,
    )

    evaluation_pipeline = EvaluationPipeline(evaluation_args)
    evaluation_pipeline.run()

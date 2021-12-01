import re

from dbpunctuator.data_process import (
    chinese_split,
    cleanup_data_from_csv,
    generate_training_data,
    remove_brackets_text,
)
from dbpunctuator.utils import DEFAULT_CHINESE_NER_MAPPING


# self defined special cleaning func
# as the ch training data used is having en puncs
def normalize_puncs(input):
    normalization = {"?": "? ", "!": "！"}
    normalizer = re.compile(
        "({})".format("|".join(map(re.escape, normalization.keys())))
    )
    return normalizer.sub(lambda m: normalization[m.string[m.start() : m.end()]], input)


def merge_data(whole_data_path, *tokens_data_paths):
    with open(whole_data_path, "w+") as whole_data_file:
        for cleaned_data_path in tokens_data_paths:
            with open(cleaned_data_path, "r") as data_file:
                whole_data_file.write(data_file.read())


if __name__ == "__main__":
    # cleaned Chinese training data
    # for ch data
    cleanup_data_from_csv(
        "./original_data/ted_talks_zh-cn.csv",
        "transcript",
        "./training_data/chinese_cleaned_text.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
        additional_to_remove=["℃", "♪"],
        special_cleaning_funcs=[normalize_puncs, chinese_split, remove_brackets_text],
    )


    generate_training_data(
        "./training_data/chinese_cleaned_text.txt",
        "./training_data/chinese_token_tag_data.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
    )

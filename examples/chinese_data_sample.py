import re

from dbpunctuator.data_process import (
    chinese_split,
    cleanup_data_from_csv,
    generate_training_data,
    remove_brackets_text,
)
from dbpunctuator.utils import CHINESE_PUNCS, DEFAULT_CHINESE_NER_MAPPING


# self defined special cleaning func
# as the ch training data used is having en puncs
def normalize_puncs(input):
    normalization = {"?": "? ", "!": "！"}
    normalizer = re.compile(
        "({})".format("|".join(map(re.escape, normalization.keys())))
    )
    return normalizer.sub(lambda m: normalization[m.string[m.start() : m.end()]], input)


if __name__ == "__main__":
    # cleaned Chinese training data

    chinese_puncs_to_rm = [
        char for char in CHINESE_PUNCS if char not in DEFAULT_CHINESE_NER_MAPPING.keys()
    ]

    # for ch data
    cleanup_data_from_csv(
        "./training_data/chinese_news.csv",
        "content",
        "./training_data/chinese_cleaned_text.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
        additional_to_remove=chinese_puncs_to_rm + ["\n", "℃"],
        special_cleaning_funcs=[normalize_puncs, chinese_split, remove_brackets_text],
    )

    generate_training_data(
        "./training_data/chinese_cleaned_text.txt",
        "./training_data/chinese_token_tag_data.txt",
        ner_mapping=DEFAULT_CHINESE_NER_MAPPING,
    )

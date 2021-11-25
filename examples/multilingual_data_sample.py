import re

from dbpunctuator.data_process import (
    chinese_split,
    cleanup_data_from_csv,
    generate_training_data,
    remove_brackets_text,
)
from dbpunctuator.utils import (
    CHINESE_PUNCS,
    DEFAULT_CHINESE_NER_MAPPING,
    DEFAULT_ENGLISH_NER_MAPPING,
)


# in training side, data will be shuffled
def merge_data(whole_data_path, *cleaned_data_paths):
    with open(whole_data_path, "w+") as whole_data_file:
        for cleaned_data_path in cleaned_data_paths:
            with open(cleaned_data_path, "r") as data_file:
                whole_data_file.write(data_file.read())


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
    # for en data
    cleanup_data_from_csv(
        "./training_data/transcripts.csv",
        "transcript",
        "./training_data/english_cleaned_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["—", "♫♫"],
        special_cleaning_funcs=[remove_brackets_text],
    )

    # merge data
    merge_data(
        "./training_data/all_cleaned_text.txt",
        "./training_data/chinese_cleaned_text.txt",
        "./training_data/english_cleaned_text.txt",
    )

    # generate training data
    ner_mappings = DEFAULT_ENGLISH_NER_MAPPING
    ner_mappings.update(DEFAULT_CHINESE_NER_MAPPING)

    generate_training_data(
        "./training_data/all_cleaned_text.txt",
        "./training_data/all_token_tag_data.txt",
        ner_mapping=ner_mappings,
    )

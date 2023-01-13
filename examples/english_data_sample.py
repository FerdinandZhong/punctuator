from itertools import zip_longest

import chardet
import pandas as pd

from punctuator.data_process import cleanup_data_from_csv, generate_corpus
from punctuator.utils import DEFAULT_ENGLISH_NER_MAPPING, remove_brackets_text


def lower_input(input):
    return input.lower()


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
    dataframe = pd.read_csv("./original_data/bbc-news-data.csv", sep="\t").dropna(
        subset=["content"]
    )
    cleanup_data_from_csv(
        dataframe,
        "content",
        "./training_data/english_cleaned_bbc_news_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["♫"],
        special_cleaning_funcs=[remove_brackets_text, lower_input],
    )

    with open("./original_data/news_summary.csv", "rb") as f:
        enc = chardet.detect(f.read())

    dataframe = pd.read_csv(
        "./original_data/news_summary.csv", encoding=enc["encoding"]
    ).dropna(subset=["text"])
    cleanup_data_from_csv(
        dataframe,
        "text",
        "./training_data/english_cleaned_news_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["♫"],
        special_cleaning_funcs=[remove_brackets_text, lower_input],
    )
    dataframe = pd.read_csv("./original_data/ted_talks_en.csv").dropna(
        subset=["transcript"]
    )
    cleanup_data_from_csv(
        dataframe,
        "transcript",
        "./training_data/english_cleaned_ted_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["♫"],
        special_cleaning_funcs=[remove_brackets_text, lower_input],
    )

    merge_data(
        "./training_data/english_cleaned_text.txt",
        "./training_data/english_cleaned_bbc_news_text.txt",
        "./training_data/english_cleaned_news_text.txt",
        "./training_data/english_cleaned_ted_text.txt",
    )

    generate_corpus(
        "./training_data/english_cleaned_text.txt",
        "./training_data/english_token_tag_data.txt",
    )

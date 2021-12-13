import re

import pandas as pd

from dbpunctuator.data_process import (
    cleanup_data_from_csv,
    generate_training_data,
    remove_brackets_text,
)
from dbpunctuator.training import ValidationArguments, ValidationPipeline
from dbpunctuator.utils import DEFAULT_ENGLISH_NER_MAPPING


def lower_input(input):
    return input.lower()


def remove_starting(input):
    regex = re.compile(r"^([^.]*)\:")
    return regex.sub("", input)


if __name__ == "__main__":
    # test with left NEWs data
    dataframe = (
        pd.read_csv("./original_data/Articles.csv", encoding="ISO-8859-1")
        .dropna(subset=["Article"])
        .sample(500)
    )
    cleanup_data_from_csv(
        dataframe,
        "Article",
        "./validation_data/english_cleaned_news_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["|", "´"],
        special_cleaning_funcs=[remove_starting, remove_brackets_text, lower_input],
    )

    generate_training_data(
        "./validation_data/english_cleaned_news_text.txt",
        "./validation_data/english_token_tag_news_data.txt",
    )

    validation_args = ValidationArguments(
        data_file_path="validation_data/english_token_tag_news_data.txt",
        model_name_or_path="Qishuai/distilbert_punctuator_en",
        tokenizer_name="Qishuai/distilbert_punctuator_en",
        min_sequence_length=100,
        max_sequence_length=200,
        batch_size=16,
    )

    validate_pipeline = ValidationPipeline(validation_args)
    validate_pipeline.run()

    # test with more ted talks
    trained_ids = pd.read_csv("original_data/ted_talks_en.csv")["talk_id"].tolist()
    dataframe = pd.read_csv("original_data/ted_talks_more.csv").dropna(
        subset=["transcript"]
    )
    dataframe = dataframe.loc[~dataframe["talk_id"].isin(trained_ids)]
    cleanup_data_from_csv(
        dataframe,
        "transcript",
        "./validation_data/english_cleaned_ted_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["♫"],
        special_cleaning_funcs=[remove_brackets_text, lower_input],
    )

    generate_training_data(
        "./validation_data/english_cleaned_ted_text.txt",
        "./validation_data/english_token_tag_ted_data.txt",
    )

    validation_args = ValidationArguments(
        data_file_path="validation_data/english_token_tag_ted_data.txt",
        model_name_or_path="Qishuai/distilbert_punctuator_en",
        tokenizer_name="Qishuai/distilbert_punctuator_en",
        min_sequence_length=100,
        max_sequence_length=200,
        batch_size=16,
    )

    validate_pipeline = ValidationPipeline(validation_args)
    validate_pipeline.run()

import re

import pandas as pd

from dbpunctuator.data_process import cleanup_data_from_csv, generate_training_data
from dbpunctuator.training import EvaluationArguments, EvaluationPipeline, generate_evaluation_data
from dbpunctuator.utils import DEFAULT_ENGLISH_NER_MAPPING, remove_brackets_text


def lower_input(input):
    return input.lower()


def remove_starting(input):
    regex = re.compile(r"^([^.]*)\:")
    return regex.sub("", input)


if __name__ == "__main__":
    # # test with left NEWs data
    # dataframe = (
    #     pd.read_csv("./original_data/Articles.csv", encoding="ISO-8859-1")
    #     .dropna(subset=["Article"])
    #     .sample(500)
    # )
    # cleanup_data_from_csv(
    #     dataframe,
    #     "Article",
    #     "./evaluation_data/english_cleaned_news_text.txt",
    #     ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
    #     additional_to_remove=["|", "´"],
    #     special_cleaning_funcs=[remove_starting, remove_brackets_text, lower_input],
    # )

    # generate_training_data(
    #     "./evaluation_data/english_cleaned_news_text.txt",
    #     "./evaluation_data/english_token_tag_news_data.txt",
    # )

    # validation_args = ValidationArguments(
    #     data_file_path="evaluation_data/english_token_tag_news_data.txt",
    #     model_name_or_path="./models/english_punctuator_rdrop",
    #     tokenizer_name="distilbert-base-uncased",
    #     min_sequence_length=100,
    #     max_sequence_length=200,
    #     batch_size=16,
    #     gpu_device=2,
    # )

    # validate_pipeline = ValidationPipeline(validation_args)
    # validate_pipeline.run()

    # test with more ted talks
    trained_ids = pd.read_csv("original_data/ted_talks_en.csv")["talk_id"].tolist()
    dataframe = pd.read_csv("original_data/ted_talks_more.csv").dropna(
        subset=["transcript"]
    )
    dataframe = dataframe.loc[~dataframe["talk__id"].isin(trained_ids)]
    cleanup_data_from_csv(
        dataframe,
        "transcript",
        "./evaluation_data/english_cleaned_ted_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["♫"],
        special_cleaning_funcs=[remove_brackets_text, lower_input],
    )

    generate_training_data(
        "./evaluation_data/english_cleaned_ted_text.txt",
        "./evaluation_data/english_token_tag_ted_data.txt",
    )

    label2id = {
        "O": 0,
        "COMMA": 1,
        "PERIOD": 2,
        "QUESTIONMARK": 3,
        "EXLAMATIONMARK": 4
    }
    evalution_corpus, evaluation_tags = generate_evaluation_data("evaluation_data/english_token_tag_ted_data.txt", 16, 256)
    evaluation_tags = [[label2id[tag] for tag in doc] for doc in evaluation_tags]
    validation_args = EvaluationArguments(
        evaluation_corpus=evalution_corpus,
        evaluation_tags=evaluation_tags,
        model_name_or_path="./models/english_punctuator_no_rdrop",
        tokenizer_name="distilbert-base-uncased",
        batch_size=16,
        gpu_device=2,
        label2id=label2id
    )

    validate_pipeline = EvaluationPipeline(validation_args)
    validate_pipeline.run()

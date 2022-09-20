import re

import pandas as pd

from dbpunctuator.data_process import cleanup_data_from_csv, generate_corpus
from dbpunctuator.training import (
    EvaluationArguments,
    EvaluationPipeline,
    generate_evaluation_data,
)
from dbpunctuator.utils import DEFAULT_ENGLISH_NER_MAPPING, remove_brackets_text


def lower_input(input):
    return input.lower()


def remove_starting(input):
    regex = re.compile(r"^([^.]*)\:")
    return regex.sub("", input)


if __name__ == "__main__":
    # test with more ted talks
    trained_ids = pd.read_csv("original_data/ted_talks_en.csv")["talk_id"].tolist()
    dataframe = pd.read_csv("original_data/TED_Talk.csv").dropna(subset=["transcript"])
    dataframe = dataframe.loc[~dataframe["talk__id"].isin(trained_ids)]
    cleanup_data_from_csv(
        dataframe,
        "transcript",
        "./evaluation_data/english_cleaned_ted_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["♫"],
        special_cleaning_funcs=[remove_brackets_text, lower_input],
    )

    generate_corpus(
        "./evaluation_data/english_cleaned_ted_text.txt",
        "./evaluation_data/english_token_tag_ted_data.txt",
    )

    # must be exact same as model's config
    label2id = {"COMMA": 2, "EXLAMATIONMARK": 1, "O": 3, "PERIOD": 4, "QUESTIONMARK": 0}
    evalution_corpus, evaluation_tags = generate_evaluation_data(
        "evaluation_data/english_token_tag_ted_data.txt", 16, 256
    )
    evaluation_tags = [[label2id[tag] for tag in doc] for doc in evaluation_tags]
    evaluation_args = EvaluationArguments(
        evaluation_corpus=evalution_corpus,
        evaluation_tags=evaluation_tags,
        model_name_or_path="Qishuai/distilbert_punctuator_en",
        tokenizer_name="Qishuai/distilbert_punctuator_en",
        batch_size=16,
        gpu_device=2,
        label2id=label2id,
    )

    evaluation_pipeline = EvaluationPipeline(evaluation_args)
    evaluation_pipeline.run()

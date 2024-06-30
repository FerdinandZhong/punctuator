import argparse

from punctuator.training.general_bert_based_pretrain import (
    PreTrainingArguments,
    PreTrainingPipeline,
)

parser = argparse.ArgumentParser()
parser = PreTrainingArguments.add_cli_args(parser)
training_args = parser.parse_args()
(
    training_corpus,
    validation_corpus,
    training_punctuation_counts,
    val_punctuation_counts,
) = PreTrainingArguments.generate_corpus(training_args)

training_args = PreTrainingArguments.from_cli_args(
    args=training_args,
    training_corpus=training_corpus,
    training_punctuation_counts=training_punctuation_counts,
    validation_corpus=validation_corpus,
    val_punctuation_counts=val_punctuation_counts,
)

training_pipeline = PreTrainingPipeline(training_args)
training_pipeline.run()

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
    training_has_punctuation_list,
    val_has_punctuation_list,
) = PreTrainingArguments.generate_corpus(training_args)

training_args = PreTrainingArguments.from_cli_args(
    args=training_args,
    training_corpus=training_corpus,
    training_has_punctuation_list=training_has_punctuation_list,
    validation_corpus=validation_corpus,
    val_has_punctuation_list=val_has_punctuation_list,
)

training_pipeline = PreTrainingPipeline(training_args)
training_pipeline.run()

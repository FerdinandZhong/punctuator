import argparse

from punctuator.training.general_ner_train_mlm_based import (
    NERTrainingArguments,
    NERTrainingPipeline,
)

parser = argparse.ArgumentParser()
parser = NERTrainingArguments.add_cli_args(parser)
training_args = parser.parse_args()
(
    training_corpus,
    validation_corpus,
    training_tags,
    validation_tags,
    label2id,
) = NERTrainingArguments.generate_corpus(training_args)

training_args = NERTrainingArguments.from_cli_args(
    args=training_args,
    training_corpus=training_corpus,
    training_tags=training_tags,
    validation_corpus=validation_corpus,
    validation_tags=validation_tags,
    label2id=label2id,
)

training_pipeline = NERTrainingPipeline(training_args)
training_pipeline.run()

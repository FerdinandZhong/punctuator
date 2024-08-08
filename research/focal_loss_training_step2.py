import argparse

from punctuator.training.general_ner_train_focal_loss_step2 import (
    Step2NERTrainingArguments,
    Step2NERTrainingPipeline,
)

parser = argparse.ArgumentParser()
parser = Step2NERTrainingArguments.add_cli_args(parser)
training_args = parser.parse_args()
(
    training_corpus,
    validation_corpus,
    training_tags,
    validation_tags,
    training_step1_features,
    validation_step1_features,
    label2id,
) = Step2NERTrainingArguments.generate_corpus(training_args)

training_args = Step2NERTrainingArguments.from_cli_args(
    args=training_args,
    training_corpus=training_corpus,
    training_tags=training_tags,
    validation_corpus=validation_corpus,
    validation_tags=validation_tags,
    training_step1_features=training_step1_features,
    validation_step1_features=validation_step1_features,
    label2id=label2id,
)

training_pipeline = Step2NERTrainingPipeline(training_args)
training_pipeline.run()

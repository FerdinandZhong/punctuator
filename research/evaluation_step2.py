import argparse

from punctuator.training.evaluation_step2 import EvaluationArguments, EvaluationPipeline

parser = argparse.ArgumentParser()
parser = EvaluationArguments.add_cli_args(parser)
evaluation_args = parser.parse_args()
(
    evaluation_corpus,
    evaluation_tags,
    evaluation_step1_features,
    label2id,
) = EvaluationArguments.generate_corpus(evaluation_args)

evaluation_args = EvaluationArguments.from_cli_args(
    args=evaluation_args,
    evaluation_corpus=evaluation_corpus,
    evaluation_tags=evaluation_tags,
    evaluation_step1_features=evaluation_step1_features,
    label2id=label2id,
)

evaluation_pipeline = EvaluationPipeline(evaluation_args)
evaluation_pipeline.run()

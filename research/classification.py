import argparse
from punctuator.training.classification import ClassificationArguments, ClassificationPipeline

parser = argparse.ArgumentParser()
parser = ClassificationArguments.add_cli_args(parser)
classification_args = parser.parse_args()
corpus, label2id = ClassificationArguments.generate_corpus(
    classification_args
)

classification_args = ClassificationArguments.from_cli_args(
    args=classification_args,
    corpus=corpus,
    label2id=label2id,
)

classification_pipeline = ClassificationPipeline(classification_args)
classification_pipeline.run()

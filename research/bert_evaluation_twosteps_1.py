import logging

from punctuator.inference import InferenceArguments
from punctuator.inference.inference_pipeline import InferencePipeline
from punctuator.training import EvaluationArguments, EvaluationPipeline, process_data
from punctuator.utils.utils import register_logger
from punctuator.utils import Models

test_data_file_path = "data/IWSLT/twosteps/step1/test2011"

with open(test_data_file_path, "r") as file:
    test_data = file.readlines()

# must be exact same as model's config
label2id = {"O": 0, "PERIOD": 1, "QUESTION": 2}
evalution_corpus, evaluation_tags = process_data(test_data, 256, 256)
evaluation_tags = [[label2id[tag] for tag in doc] for doc in evaluation_tags]
evaluation_args = EvaluationArguments(
    evaluation_corpus=evalution_corpus,
    evaluation_tags=evaluation_tags,
    model=Models.BERT,
    model_weight_name="models/iwslt_bert_finetune_rdrop_twosteps_1",
    tokenizer_name="bert-large-uncased",
    batch_size=16,
    gpu_device=0,
    label2id=label2id,
)

evaluation_pipeline = EvaluationPipeline(evaluation_args)
evaluation_pipeline.run()
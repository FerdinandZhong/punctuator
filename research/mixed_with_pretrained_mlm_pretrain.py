import argparse

from punctuator.mlm_pretraining import (
    PretrainingArguments,
    PretrainingPipeline,
    process_data,
)
from punctuator.utils import Models

parser = argparse.ArgumentParser()
parser.add_argument(
    "--last_layer", help="last layer of the output directory", default=""
)
args = parser.parse_args()

training_data_file_path = "data/IWSLT/formatted/train2012"
eval_data_file_path = "data/IWSLT/formatted/dev2012"

with open(training_data_file_path, "r") as file:
    training_raw = file.readlines()

with open(eval_data_file_path, "r") as file:
    val_raw = file.readlines()


(
    training_corpus,
    training_span_labels,
) = process_data(training_raw, 128)

(
    validation_corpus,
    validation_span_labels,
) = process_data(val_raw, 128)

training_args = PretrainingArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_span_labels=training_span_labels,
    validation_span_labels=validation_span_labels,
    plm_model=Models.BERT,
    plm_model_config_name="bert-large-uncased",
    tokenizer_name="bert-large-uncased",
    epoch=50,
    batch_size=48,
    model_storage_dir=f"models/pretraining_mlm/mixed_with_pretrained/bert_large_uncased/{args.last_layer}",
    early_stop_count=5,
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    warm_up_steps=1000,
    tensorboard_log_dir=f"runs/pretraining_mlm/mixed_with_pretrained/bert_large_uncased/pretrain/{args.last_layer}",
    span_only=False,
    mask_rate=0.15,
    gpu_device=0,
    load_weight=True,
)
training_pipeline = PretrainingPipeline(training_args, verbose=True)

training_pipeline.tokenize().static_mask().generate_dataset().train().persist()

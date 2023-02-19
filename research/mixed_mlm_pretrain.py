from punctuator.mlm_pretraining import (
    PretrainingArguments,
    PretrainingPipeline,
    process_data,
)
from punctuator.utils import Models

training_data_file_path = "data/IWSLT/formatted/train2012"
eval_data_file_path = "data/IWSLT/formatted/dev2012"

with open(training_data_file_path, "r") as file:
    training_raw = file.readlines()

with open(eval_data_file_path, "r") as file:
    val_raw = file.readlines()


(
    training_corpus,
    training_span_labels,
) = process_data(training_raw, 64, 256)

(
    validation_corpus,
    validation_span_labels,
) = process_data(val_raw, 64, 256)

training_args = PretrainingArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_span_labels=training_span_labels,
    validation_span_labels=validation_span_labels,
    plm_model=Models.BERT,
    plm_model_config_name="bert-large-uncased",
    tokenizer_name="bert-large-uncased",
    epoch=50,
    batch_size=16,
    model_storage_dir="models/pretraining_mlm/mixed/bert_large_uncased",
    early_stop_count=5,
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    warm_up_steps=1000,
    tensorboard_log_dir="runs/pretraining_mlm/mixed/bert_large_uncased/pretrain",
    span_only=False,
    mask_rate=0.15,
    gpu_device=0,
)
training_pipeline = PretrainingPipeline(training_args, verbose=True)

training_pipeline.tokenize().mask().generate_dataset().train().persist()

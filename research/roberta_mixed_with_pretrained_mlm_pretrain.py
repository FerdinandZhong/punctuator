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
) = process_data(training_raw, 64, 128, is_return_list=True)

(
    validation_corpus,
    validation_span_labels,
) = process_data(val_raw, 64, 128, is_return_list=True)

training_args = PretrainingArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_span_labels=training_span_labels,
    validation_span_labels=validation_span_labels,
    plm_model=Models.ROBERTA,
    plm_model_config_name="roberta-large",
    tokenizer_name="roberta-large",
    epoch=50,
    batch_size=96,
    model_storage_dir="models/pretraining_mlm/mixed_with_pretrained/roberta-large/prefix_space",
    early_stop_count=5,
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    warm_up_steps=2000,
    tensorboard_log_dir="runs/pretraining_mlm/mixed_with_pretrained/roberta-large/pretrain/prefix_space",
    span_only=False,
    is_dynamic_mask=True,
    mask_rate=0.15,
    gpu_device=0,
    load_weight=True,
    additional_tokenizer_config={"add_prefix_space": True},
)
training_pipeline = PretrainingPipeline(training_args, verbose=True)

training_pipeline.tokenize(
    is_split_into_words=True
).generate_dataset().train().persist()


"""
Pretraining 1: not split into tokens and tokenization without prefix_space (not shuffle)
Pretraining 2: split into tokens and tokenization with prefix_space (shuffle) based on pretraining 1
"""

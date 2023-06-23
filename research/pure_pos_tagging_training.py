from punctuator.training import process_data
from punctuator.training.pos_ner_train import (
    PosNERTrainingArguments,
    PosNERTrainingPipeline,
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
    training_tags,
) = process_data(training_raw, 256, 256)

(
    validation_corpus,
    validation_tags,
) = process_data(val_raw, 256, 256)

label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
training_tags = [[label2id[tag] for tag in doc] for doc in training_tags]
validation_tags = [[label2id[tag] for tag in doc] for doc in validation_tags]

training_args = PosNERTrainingArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_tags=training_tags,
    validation_tags=validation_tags,
    model=Models.BERT_TOKEN_CLASSIFICATION,
    model_weight_name="bert-large-uncased",
    tokenizer_name="bert-large-uncased",
    epoch=40,
    batch_size=16,
    model_storage_dir="models/iwslt_pure_pos_tagging",
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    warm_up_steps=1000,
    r_drop=False,
    r_alpha=0.5,
    tensorboard_log_dir="runs/iwslt_pure_pos_tagging",
    label2id=label2id,
    early_stop_count=5,
    gpu_device=2,
    training_pos_tagging_path="data/IWSLT/formatted/postagging_train",
    val_pos_tagging_path="data/IWSLT/formatted/postagging_val",
)

training_pipeline = PosNERTrainingPipeline(training_args)
training_pipeline.run()
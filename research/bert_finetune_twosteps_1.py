from punctuator.training import (
    NERTrainingArguments,
    NERTrainingPipeline,
    process_data,
)
from punctuator.utils import (
    Models
)

training_data_file_path = "data/IWSLT/twosteps/step1/train2012"
eval_data_file_path = "data/IWSLT/twosteps/step1/dev2012"

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

label2id = {"O": 0, "PERIOD": 1, "QUESTION": 2}
training_tags = [[label2id[tag] for tag in doc] for doc in training_tags]
validation_tags = [[label2id[tag] for tag in doc] for doc in validation_tags]

training_args = NERTrainingArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_tags=training_tags,
    validation_tags=validation_tags,
    model=Models.BERT,
    model_weight_name="bert-large-uncased",
    tokenizer_name="bert-large-uncased",
    epoch=40,
    batch_size=16,
    model_storage_dir="models/iwslt_bert_finetune_rdrop_twosteps_1",
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    gpu_device=0,
    warm_up_steps=1000,
    r_drop=True,
    r_alpha=0.5,
    tensorboard_log_dir="runs/iwslt_bert_finetune_rdrop_twosteps_1",
    label2id=label2id,
    early_stop_count=5,
)

training_pipeline = NERTrainingPipeline(training_args)
training_pipeline.run()

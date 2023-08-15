import os
from datetime import datetime

from punctuator.pointer_network import (
    Stage1TrainingArguments,
    Stage1TrainingPipeline,
    generate_stage1_data,
)
from punctuator.utils import Models

current_date = datetime.now().strftime("%Y-%m-%d")
model_storage_dir = f"models/pointer_network/stage1/method1/{current_date}/"
tensorboard_log_dir = f"runs/pointer_network/stage1/method1/{current_date}/"

os.makedirs(model_storage_dir, exist_ok=True)
os.makedirs(tensorboard_log_dir, exist_ok=True)

training_data_file_path = "data/IWSLT/formatted/train2012"
eval_data_file_path = "data/IWSLT/formatted/dev2012"

with open(training_data_file_path, "r") as file:
    training_raw = file.readlines()

with open(eval_data_file_path, "r") as file:
    val_raw = file.readlines()


(
    training_corpus,
    training_tags,
) = generate_stage1_data(training_raw)

(
    validation_corpus,
    validation_tags,
) = generate_stage1_data(val_raw)

training_args = Stage1TrainingArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_tags=training_tags,
    validation_tags=validation_tags,
    pretrained_model=Models.BART,
    model_config_name="models/pointer_network/stage1/method1/2023-06-29",
    tokenizer_name="facebook/bart-large",
    model_name="models/pointer_network/stage1/method1/2023-06-29",
    load_local_model=True,
    max_input_length=640,
    pointer_tolerance=2,
    epoch=4,
    batch_size=48,
    model_storage_dir=model_storage_dir,
    early_stop_count=5,
    additional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    additional_tokenizer_config={"add_prefix_space": True},
    warm_up_steps=1000,
    tensorboard_log_dir=tensorboard_log_dir,
    gpu_device=0,
)
training_pipeline = Stage1TrainingPipeline(training_args, verbose=True)

training_pipeline.train().persist()
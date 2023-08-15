import os
from datetime import datetime

from punctuator.pointer_network import (
    Stage1TrainingBatchedArguments,
    Stage1TrainingBatchedPipeline,
    generate_batched_stage1_data,
)
from punctuator.utils import Models

current_date = datetime.now().strftime("%Y-%m-%d")
model_storage_dir = f"models/pointer_network/stage1/method1_batched/{current_date}/"
tensorboard_log_dir = f"runs/pointer_network/stage1/method1_batched/{current_date}/"

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
) = generate_batched_stage1_data(training_raw, 320)

print(f"length of training_corpus sample: {len(training_corpus[0])}")
(
    validation_corpus,
    validation_tags,
) = generate_batched_stage1_data(val_raw, 320)

training_args = Stage1TrainingBatchedArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_tags=training_tags,
    validation_tags=validation_tags,
    pretrained_model=Models.BART,
    model_config_name="facebook/bart-large",
    tokenizer_name="facebook/bart-large",
    model_name="facebook/bart-large",
    pointer_tolerance=2,
    epoch=10,
    batch_size=12,
    model_storage_dir=model_storage_dir,
    early_stop_count=5,
    additional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    additional_tokenizer_config={"add_prefix_space": True},
    warm_up_steps=1000,
    tensorboard_log_dir=tensorboard_log_dir,
    gpu_device=0,
)
training_pipeline = Stage1TrainingBatchedPipeline(training_args, verbose=False)

training_pipeline.train().persist()

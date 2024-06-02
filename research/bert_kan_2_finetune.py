import argparse

from punctuator.training import NERTrainingArguments, NERTrainingPipeline, process_data
from punctuator.utils import Models

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model_storage_dir",
    help="The storage directory of the finetuned model",
    default="models/",
)
parser.add_argument(
    "--tensorboard_log_dir",
    help="The tensorboard directory of the finetuning",
    default="runs/",
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
    training_tags,
) = process_data(training_raw, 128)

(
    validation_corpus,
    validation_tags,
) = process_data(val_raw, 128)

label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
training_tags = [[label2id[tag] for tag in doc] for doc in training_tags]
validation_tags = [[label2id[tag] for tag in doc] for doc in validation_tags]

training_args = NERTrainingArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_tags=training_tags,
    validation_tags=validation_tags,
    model=Models.BERT_KAN_2,
    load_backbone_only=True,
    model_weight_name="bert-large-uncased",
    tokenizer_name="bert-large-uncased",
    epoch=40,
    batch_size=16,
    model_storage_dir=args.model_storage_dir,
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    gpu_device=0,
    warm_up_steps=1000,
    r_drop=False,
    r_alpha=0.5,
    tensorboard_log_dir=args.tensorboard_log_dir,
    label2id=label2id,
    early_stop_count=5,
)

training_pipeline = NERTrainingPipeline(training_args)
training_pipeline.run()

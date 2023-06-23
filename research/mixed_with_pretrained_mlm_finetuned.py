import argparse

from punctuator.mlm_pretraining import FineTuneArguments, FinetunePipeline
from punctuator.training import process_data
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
    training_tags,
) = process_data(training_raw, 128)

(
    validation_corpus,
    validation_tags,
) = process_data(val_raw, 128)

label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
training_tags = [[label2id[tag] for tag in doc] for doc in training_tags]
validation_tags = [[label2id[tag] for tag in doc] for doc in validation_tags]

training_args = FineTuneArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_tags=training_tags,
    validation_tags=validation_tags,
    plm_model=Models.BERT_TOKEN_CLASSIFICATION,
    plm_path=f"models/pretraining_mlm/mixed_with_pretrained/bert_large_uncased/{args.last_layer}",
    model_storage_dir=f"models/pretraining_mlm/mixed_with_pretrained/bert_large_uncased/{args.last_layer}",
    tokenizer_name="bert-large-uncased",
    model_weight_name="bert-large-uncased",
    model_config_name="bert-large-uncased",
    label2id=label2id,
    epoch=40,
    batch_size=64,
    early_stop_count=0,
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    warm_up_steps=1000,
    tensorboard_log_dir=f"runs/pretraining_mlm/mixed_with_pretrained/bert_large_uncased/finetune/{args.last_layer}",
    r_drop=False,
    r_alpha=0.5,
    gpu_device=0,
)
training_pipeline = FinetunePipeline(training_args, verbose=True)

training_pipeline.tokenize().generate_dataset().fine_tune().persist()
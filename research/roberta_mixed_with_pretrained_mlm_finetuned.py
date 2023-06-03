from punctuator.mlm_pretraining import FineTuneArguments, FinetunePipeline
from punctuator.training import process_data
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
) = process_data(training_raw, 64, 128, is_return_list=True)

(
    validation_corpus,
    validation_tags,
) = process_data(val_raw, 64, 128, is_return_list=True)

label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
training_tags = [[label2id[tag] for tag in doc] for doc in training_tags]
validation_tags = [[label2id[tag] for tag in doc] for doc in validation_tags]

training_args = FineTuneArguments(
    training_corpus=training_corpus,
    validation_corpus=validation_corpus,
    training_tags=training_tags,
    validation_tags=validation_tags,
    model=Models.ROBERTA_TOKEN_CLASSIFICATION,
    plm_path="models/pretraining_mlm/mixed_with_pretrained/roberta-large/prefix_space_further",
    model_storage_dir="models/pretraining_mlm/mixed_with_pretrained/roberta-large/prefix_space_further",
    tokenizer_name="roberta-large",
    model_config_name="roberta-large",
    label2id=label2id,
    epoch=40,
    batch_size=64,
    early_stop_count=0,
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    warm_up_steps=1000,
    tensorboard_log_dir="runs/pretraining_mlm/mixed_with_pretrained/roberta-large/finetune",
    r_drop=False,
    r_alpha=0.5,
    gpu_device=0,
    additional_tokenizer_config={"add_prefix_space": True},
)
training_pipeline = FinetunePipeline(training_args, verbose=True)

training_pipeline.tokenize().generate_dataset().fine_tune().persist()

from punctuator.nsp_training import TrainingArguments, TrainingPipeline
from punctuator.utils import Models

training_data_path = "data/IWSLT/multi_label_nsp/bert_large_uncased/train2012.json"
validation_data_path = "data/IWSLT/multi_label_nsp/bert_large_uncased/dev2012.json"

# with open(training_data_path, "r") as json_file:
#     training_source_data = json.load(json_file)

# with open(validation_data_path, "r") as json_file:
#     validation_source_data = json.load(json_file)

label2id = {"O": 0, "COMMA": 1, "PERIOD": 2, "QUESTION": 3}
training_args = TrainingArguments(
    training_data_path=training_data_path,
    validation_data_path=validation_data_path,
    plm_model=Models.BERT,
    plm_model_weight_name="bert-large-uncased",
    tokenizer_name="bert-large-uncased",
    target_sequence_length=200,
    lower_boundary=5,
    epoch=40,
    batch_size=64,
    model_storage_dir="models/multi_label_nsp/bert_large_uncased",
    addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
    warm_up_steps=1000,
    r_drop=False,
    tensorboard_log_dir="runs/multi_label_nsp/bert_large_uncased",
    label2id=label2id,
    early_stop_count=5,
    gpu_device=0,
)
training_pipeline = TrainingPipeline(training_args, verbose=True)

training_pipeline.tokenize().generate_dataset().train().persist()

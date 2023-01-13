from punctuator.training import (
    NERTrainingArguments,
    NERTrainingPipeline,
    generate_training_data_splitting,
)

if __name__ == "__main__":
    data_file_path = "training_data/chinese_token_tag_data.txt"

    (
        training_corpus,
        training_tags,
        validation_corpus,
        validation_tags,
    ) = generate_training_data_splitting(data_file_path, 16, 256, 0.25)

    label2id = {
        "O": 0,
        "C_COMMA": 1,
        "C_PERIOD": 2,
        "C_QUESTIONMARK": 3,
        "C_EXLAMATIONMARK": 4,
        "C_DUNHAO": 5,
    }

    training_tags = [[label2id[tag] for tag in doc] for doc in training_tags]
    validation_tags = [[label2id[tag] for tag in doc] for doc in validation_tags]

    training_args = NERTrainingArguments(
        training_corpus=training_corpus,
        validation_corpus=validation_corpus,
        training_tags=training_tags,
        validation_tags=validation_tags,
        model_weight_name="models/distilbert-base-chinese",
        tokenizer_name="bert-base-chinese",
        epoch=20,
        batch_size=16,
        model_storage_dir="models/chinese_punctuator_no_rdrop_new",
        addtional_model_config={"dropout": 0.3, "attention_dropout": 0.3},
        gpu_device=0,
        warm_up_steps=500,
        r_drop=False,
        r_alpha=0.2,
        tensorboard_log_dir="runs/chinese_punctuator_rdrop",
        label2id=label2id,
        early_stop_count=4,
    )

    training_pipeline = NERTrainingPipeline(training_args)
    training_pipeline.run()

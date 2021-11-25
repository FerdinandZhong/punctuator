from dbpunctuator.training import TrainingArguments, TrainingPipeline
from dbpunctuator.utils.utils import register_logger

if __name__ == "__main__":
    register_logger()

    training_args = TrainingArguments(
        data_file_path="training_data/chinese_token_tag_data.txt",
        model_name="models/distilbert-base-chinese",
        tokenizer_name="bert-base-chinese",
        split_rate=0.2,
        sequence_length=100,
        epoch=5,
        batch_size=64,
        model_storage_dir="models/chinese_punctuator",
        tag2id_storage_name="tag2id.json",
        addtional_model_config={"dropout": 0.2},
    )

    training_pipeline = TrainingPipeline(training_args)
    training_pipeline.run()
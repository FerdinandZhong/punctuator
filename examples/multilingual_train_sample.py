from dbpunctuator.training import TrainingArguments, TrainingPipeline
from dbpunctuator.utils.utils import register_logger

if __name__ == "__main__":
    register_logger()

    training_args = TrainingArguments(
        data_file_path="training_data/all_token_tag_data.txt",
        model_name="distilbert-base-multilingual-cased",
        tokenizer_name="distilbert-base-multilingual-cased",
        split_rate=0.3,
        sequence_length=120,
        epoch=5,
        batch_size=64,
        model_storage_dir="models/multilingual_punctuator",
        tag2id_storage_name="tag2id.json",
        addtional_model_config={"dropout": 0.3},
    )

    training_pipeline = TrainingPipeline(training_args)
    training_pipeline.run()

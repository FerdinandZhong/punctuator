from dbpunctuator.data_process import (
    cleanup_data_from_csv,
    generate_training_data,
    remove_brackets_text,
)
from dbpunctuator.utils import DEFAULT_ENGLISH_NER_MAPPING

if __name__ == "__main__":
    cleanup_data_from_csv(
        "./original_data/ted_talks_en.csv",
        "transcript",
        "./training_data/english_cleaned_text.txt",
        ner_mapping=DEFAULT_ENGLISH_NER_MAPPING,
        additional_to_remove=["♫"],
        special_cleaning_funcs=[remove_brackets_text],
    )

    generate_training_data(
        "./training_data/english_cleaned_text.txt",
        "./training_data/english_token_tag_data.txt",
    )

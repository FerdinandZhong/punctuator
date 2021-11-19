from dbpunctuator.data_process import (
    cleanup_data_from_csv,
    generate_training_data,
    remove_brackets_text,
)

if __name__ == "__main__":
    cleanup_data_from_csv(
        "./training_data/transcripts.csv",
        "transcript",
        "./training_data/cleaned_text.txt",
        additional_to_remove=["-"],
        special_cleaning_funcs=[remove_brackets_text],
    )
    generate_training_data(
        "./training_data/cleaned_text.txt", "./training_data/all_token_tag_data.txt"
    )

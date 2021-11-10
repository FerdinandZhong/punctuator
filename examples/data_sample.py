from data_process import cleanup_data_from_csv, generate_training_data

if __name__ == "__main__":
    cleanup_data_from_csv(
        "./training_data/transcripts.csv",
        "transcript",
        "./sample_data/cleaned_text.txt",
    )
    generate_training_data(
        "./sample_data/cleaned_text.txt", "./sample_data/all_token_tag_data.txt"
    )

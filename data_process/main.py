import pandas as pd
from tqdm import tqdm
from data_process.data_cleanning import (
    dataframe_data_cleaning,
    remove_brackets_text,
)

ner_mapping = {
    ",": "C",
    ".": "P",
    "?": "Q",
    "!": "E"
}

def generate_training_data_from_csv(csv_path, target_col, output_file_path):
    dataframe = pd.read_csv(csv_path)
    print(dataframe.head())
    additional_to_remove = ["—", "..."]
    kept_punctuations = set(ner_mapping.keys())
    result_df = dataframe_data_cleaning(
        dataframe[:1500],
        target_col,
        kept_punctuations,
        additional_to_remove,
        remove_brackets_text,
    )
    print(result_df.head())
    with open(output_file_path, "w+") as output_file:
        for row in result_df[target_col].tolist():
            output_file.write("%s. \n" % row)

def process_line(line):
    text_list = line.split()
    word_list = []
    token_list = []
    latest_word = text_list.pop(0)
    while latest_word in ner_mapping:
        if not text_list:
            break
        latest_word = text_list.pop(0)
    latest_token = "O"
    latest_is_punc = False
    for word in text_list:
        if word in ner_mapping:
            if not latest_is_punc:
                latest_token = ner_mapping[word]
                latest_is_punc = True
                word_list.append(latest_word)
                token_list.append(latest_token)
            else:
                pass
        else:
            if not latest_is_punc:
                word_list.append(latest_word)
                token_list.append(latest_token)
            latest_is_punc = False
            latest_word = word
            latest_token = "O"
    if not latest_is_punc:
        word_list.append(latest_word)
        token_list.append(latest_token)
    return word_list, token_list



def generate_training_data(cleaned_data_path, training_data_path):
    with open(cleaned_data_path, "r") as data_file:
        lines = data_file.readlines()
    with open(training_data_path, "w+") as training_data_file:
        pbar = tqdm(lines)
        for line in pbar:
            words, tokens = process_line(line)
            for word, token in zip(words, tokens):
                training_data_file.write("%s   %s\n" % (word, token))


if __name__ == "__main__":
    generate_training_data_from_csv(
        "./training_data/transcripts.csv",
        "transcript",
        "./training_data/cleaned_text.txt",
    )
    generate_training_data("./training_data/cleaned_text.txt", "./training_data/training.txt")

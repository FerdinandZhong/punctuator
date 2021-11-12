import pandas as pd
from tqdm import tqdm

from data_process.data_cleanning import dataframe_data_cleaning
from utils.constant import DEFAULT_NER_MAPPING, DIGIT_MASK


def cleanup_data_from_csv(
    csv_path,
    target_col,
    output_file_path,
    ner_mapping=DEFAULT_NER_MAPPING,
    additional_to_remove=[],
    special_cleaning_funcs=[],
):
    """clean up training data from csv file

    Args:
        csv_path (string): path of training data csv
        target_col (string): column of training data in csv
        output_file_path (string): path of cleaned data
        ner_mapping (dict, optional): NER mapping of punctuation marks. Defaults to utils.constant.DEFAULT_NER_MAPPING
        additional_to_remove (list, optional): additional special characters to remove, default []
        special_cleaning_funcs (List[func], optional): additional cleaning funcs to apply to csv data, default []
    """
    dataframe = pd.read_csv(csv_path)
    additional_to_remove = ["—"]
    kept_punctuations = set(ner_mapping.keys())
    result_df = dataframe_data_cleaning(
        dataframe[1500:],
        target_col,
        kept_punctuations,
        additional_to_remove,
        *special_cleaning_funcs,
    )
    with open(output_file_path, "w+") as output_file:
        for row in result_df[target_col].tolist():
            if row:
                if row[-1] not in ner_mapping:
                    output_file.write("%s. \n" % row)
                else:
                    output_file.write("%s \n" % row)


def process_line(line, ner_mapping=DEFAULT_NER_MAPPING):
    text_list = line.split()
    word_list = []
    token_list = []
    # clean up puncs in the beginning of the text
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
            if word.isdigit():
                word = DIGIT_MASK
            latest_word = word
            latest_token = "O"
    if not latest_is_punc:
        word_list.append(latest_word)
        token_list.append(latest_token)
    return word_list, token_list


def generate_training_data(cleaned_data_path, training_data_path):
    """generate "token tag" format training data based on cleaned text

    Args:
        cleaned_data_path (string): path of cleaned data
        training_data_path (string): path of generated training data
    """
    with open(cleaned_data_path, "r") as data_file:
        lines = data_file.readlines()
    with open(training_data_path, "w+") as training_data_file:
        pbar = tqdm(lines)
        for line in pbar:
            words, tokens = process_line(line)
            for word, token in zip(words, tokens):
                training_data_file.write("%s\t%s\n" % (word, token))
        pbar.close()

import os
from pathlib import Path

from sklearn.model_selection import train_test_split

def split_dataset(token_docs, tag_docs, split_rate=0.2):
    train_texts, val_texts, train_tags, val_tags = train_test_split(token_docs, tag_docs, test_size=.2)

def read_data(file_path):
    def read_line(text_line):
        return text_line.strip().split("\t")

    token_docs = []
    tag_docs = []
    with open(file_path, "r") as data_file:
        for line in data_file:
            processed_line = read_line(line)
            token_docs.append(processed_line[0])
            tag_docs.append(processed_line[1])

    return token_docs, tag_docs

def generate_tag_ids(tag_docs):
    unique_tags = set(tag_docs)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    return tag2id, id2tag

if __name__ == "__main__":
    print(read_data("./training_data/all_token_tag_data.txt")[0][10])
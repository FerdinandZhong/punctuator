import random
from typing import List

from tqdm import tqdm

PAD_TOKEN = "[PAD]"
NORMAL_TOKEN_TAG = "O"


def read_data(file_path, sequence_length) -> List[List]:
    def read_line(text_line):
        return text_line.strip().split("\t")

    token_docs = []
    tag_docs = []
    line_index = 0

    token_doc = []
    tag_doc = []
    with open(file_path, "r") as data_file:
        pbar = tqdm(data_file.readlines())
        for line in pbar:
            if line_index == 0:
                token_doc = []
                tag_doc = []
            processed_line = read_line(line)
            token_doc.append(processed_line[0])
            tag_doc.append(processed_line[1])
            line_index += 1
            if line_index == sequence_length:
                token_docs.append(token_doc)
                tag_docs.append(tag_doc)
                line_index = 0
                pbar.update(sequence_length)
        token_doc += [PAD_TOKEN] * (sequence_length - line_index)
        tag_doc += [NORMAL_TOKEN_TAG] * (sequence_length - line_index)
        token_docs.append(token_doc)
        tag_docs.append(tag_doc)

        pbar.close()

    return token_docs, tag_docs


def generate_tag_ids(tag_docs):
    unique_tags = set([tag for tags in tag_docs for tag in tags])
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}

    return tag2id, id2tag


def train_test_split(tokens, tags, test_size=0.2, shuffle=True):
    if shuffle:
        random.shuffle(tokens)
        random.shuffle(tags)
    index = round(len(tokens) * (1 - test_size))
    return tokens[:index], tags[:index], tokens[index:], tags[index:]

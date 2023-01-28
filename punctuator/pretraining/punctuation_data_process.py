import logging
from typing import List

from tqdm import tqdm

from punctuator.utils import NORMAL_TOKEN_TAG

SPACE_TOKEN = "[SPACE]"
PUNCT_TOKEN = "[PUNCT]"
SENTENCE_ENDINGS = ["PERIOD", "QUESTION"]
logger = logging.getLogger(__name__)


def _read_data(source_data, target_sequence_length) -> List[List]:
    def read_line(text_line):
        return text_line.strip().split("\t")

    token_docs = []
    span_label_docs = []

    token_doc = []
    span_label_doc = []
    if isinstance(source_data, List):
        pbar = tqdm(source_data)
    else:
        with open(source_data, "r") as data_file:
            pbar = tqdm(data_file.readlines())
    for index, line in enumerate(pbar):
        processed_line = read_line(line)
        try:
            assert len(processed_line) == 2, "bad line"
            token_doc.append(processed_line[0])
            if processed_line[1] == NORMAL_TOKEN_TAG:
                token_doc.append(SPACE_TOKEN)
                span_label_doc.append(0)
            else:
                token_doc.append(PUNCT_TOKEN)
                span_label_doc.append(1)
        except AssertionError:
            logger.warning(f"ignore the bad line: {line}, index: {index}")
            continue
        if (
            len(token_doc) >= target_sequence_length
            and processed_line[1] in SENTENCE_ENDINGS
        ):
            token_docs.append(token_doc)
            span_label_docs.append(span_label_doc)
            pbar.update(len(token_doc))
            token_doc = []
            span_label_doc = []
    token_docs.append(token_doc)
    span_label_docs.append(span_label_doc)
    pbar.update(len(token_doc))

    pbar.close()

    return token_docs, span_label_docs


def process_data(source_data, target_sequence_length):
    """
    Function for generation of tokenized corpus and relevant tags

    Args:
        source_data(str or List): path of input data or input data
        target_sequence_length(int): target sequence length
    """
    logger.info("load data")
    texts, span_labels = _read_data(source_data, target_sequence_length)
    return texts, span_labels

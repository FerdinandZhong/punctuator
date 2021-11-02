import json
import logging
from itertools import filterfalse

import numpy as np
import torch
from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
)

from utils.constant import DIGIT_MASK, TAG_PUNCTUATOR_MAP
from utils.utils import register_logger

logger = logging.getLogger(__name__)

with open("models/tag2id.json", "r") as fp:
    tag2id = json.load(fp)
    id2tag = {id: tag for tag, id in tag2id.items()}


def process_results(device, logits, marks, all_tokens, digit_indexes):
    if device.type == "cuda":
        max_preds = logits.argmax(dim=2).detach().cpu().numpy()
    else:
        max_preds = logits.argmax(dim=2).detach().numpy()

    reduce_ignored_marks = marks >= 0

    next_upper = True
    result_texts = []
    for pred, reduce_ignored, tokens, digit_index in zip(
        max_preds, reduce_ignored_marks, all_tokens, digit_indexes
    ):
        true_pred = pred[reduce_ignored]
        result_text = ""
        for id, (index, token) in zip(true_pred, enumerate(tokens)):
            tag = id2tag[id]
            if index in digit_index:
                token = digit_index[index]
            if next_upper:
                token = token.capitalize()
            punctuator, next_upper = TAG_PUNCTUATOR_MAP[tag]
            result_text += token + punctuator
        result_texts.append(result_text.strip())
    print(result_texts)


def mark_ignored_tokens(offset_mapping):
    samples = []
    for sample_offset in offset_mapping:
        # create an empty array of -100
        sample_marks = np.ones(len(sample_offset), dtype=int) * -100
        arr_offset = np.array(sample_offset)

        # set labels whose first offset position is 0 and the second is not 0, only special tokens second is also 0
        sample_marks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = 0
        samples.append(sample_marks.tolist())

    return np.array(samples)


def preprocess(inputs):
    digit_indexes = []
    all_tokens = []
    for input in inputs:
        input_tokens = input.split()
        digits = dict(
            list(filterfalse(lambda x: not x[1].isdigit(), enumerate(input_tokens)))
        )
        for index_key in digits.keys():
            input_tokens[index_key] = DIGIT_MASK
        digit_indexes.append(digits)
        all_tokens.append(input_tokens)

    return all_tokens, digit_indexes


if __name__ == "__main__":
    register_logger()
    test_texts = [
        # ['this', 'is', 'our', 'life', 'with', 'bees', 'and', 'this', 'is', 'our', 'life', 'without', 'bees', 'bees', 'are', 'the', 'most', 'important', 'pollinators', 'of', 'our', 'fruits', 'and', 'vegetables', 'and', 'flowers', 'and', 'crops', 'like', 'alfalfa', 'hay', 'that', 'feed', 'our', 'farm', 'animals', 'more', 'than', 'one', 'third', 'of', 'the', 'worlds', 'crop', 'production', 'is', 'dependent', 'on', 'bee', 'pollination', 'but', 'the', 'ironic', 'thing', 'is', 'that', 'bees', 'are', 'not', 'out', 'there', 'pollinating', 'our', 'food', 'intentionally', 'theyre', 'out', 'there', 'because', 'they', 'need', 'to', 'eat', 'bees', 'get', 'all', 'of', 'the', 'protein', 'they', 'need', 'in', 'their', 'diet', 'from', 'pollen', 'and', 'all', 'of', 'the', 'carbohydrates', 'they', 'need', 'from', 'nectar', 'theyre', 'flowerfeeders', 'and', 'as', 'they', 'move', 'from', 'flower', 'to', 'flower', 'basically', 'on', 'a', 'shopping', 'trip', 'at', 'the', 'local', 'floral', 'mart', 'they', 'end', 'up', 'providing', 'this'],
        # ['valuable', 'pollination', 'service', 'in', 'parts', 'of', 'the', 'world', 'where', 'there', 'are', 'no', 'bees', 'or', 'where', 'they', 'plant', 'varieties', 'that', 'are', 'not', 'attractive', 'to', 'bees', 'people', 'are', 'paid', 'to', 'do', 'the', 'business', 'of', 'pollination', 'by', 'hand', 'these', 'people', 'are', 'moving', 'pollen', 'from', 'flower', 'to', 'flower', 'with', 'a', 'paintbrush', 'now', 'this', 'business', 'of', 'hand', 'pollination', 'is', 'actually', 'not', 'that', 'uncommon', 'tomato', 'growers', 'often', 'pollinate', 'their', 'tomato', 'flowers', 'with', 'a', 'handheld', 'vibrator', 'now', 'this', 'ones', 'the', 'tomato', 'tickler', 'now', 'this', 'is', 'because', 'the', 'pollen', 'within', 'a', 'tomato', 'flower', 'is', 'held', 'very', 'securely', 'within', 'the', 'male', 'part', 'of', 'the', 'flower', 'the', 'anther', 'and', 'the', 'only', 'way', 'to', 'release', 'this', 'pollen', 'is', 'to', 'vibrate', 'it', 'so', 'bumblebees', 'are', 'one', 'of', 'the', 'few', 'kinds', 'of', 'bees']
        # ["how", "are", "you", "my", "new", "number", "is", "<num>"],
        # ["can", "i", "have", "your", "phone", "number"],
        "how are you its been ten years since we met in shanghai im really happen to meet you again whats your current phone number",
        "my number is 82732212",
    ]

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    config = DistilBertConfig.from_pretrained("models/punctuator")
    model = DistilBertForTokenClassification.from_pretrained(
        "models/punctuator", config=config
    )
    model.eval()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    test_tokens, digit_indexes = preprocess(test_texts)

    inputs = tokenizer(
        test_tokens,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt",
    )

    marks = mark_ignored_tokens(inputs["offset_mapping"])

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    outputs = model(input_ids, attention_mask)

    process_results(device, outputs.logits, marks, test_tokens, digit_indexes)

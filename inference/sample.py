from transformers import (
    AdamW,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
)
from pydantic import BaseModel
import torch
from utils.utils import register_logger
import logging
from training.dataset import read_data
import numpy as np
import json


logger = logging.getLogger(__name__)

with open("models/tag2id.json", "r") as fp:
    tag2id = json.load(fp)
    id2tag = {id: tag for tag, id in tag2id.items()}

tag_punctuator_map = {
    "O": (" ", False),
    "C": (", ", False),
    "P": (". ", True),
    "Q": ("? ", True),
    "E": ("! ", True)
}


def process_results(device, logits, attention_mask, marks, all_tokens):
    if device.type == "cuda":
        max_preds = logits.argmax(dim=2).detach().cpu().numpy().flatten()
        flattened_attention = attention_mask.detach().cpu().numpy().flatten()
    else:
        max_preds = logits.argmax(dim=2).detach().numpy().flatten()
        flattened_attention = attention_mask.detach().numpy().flatten()

    reduce_ignored = marks >= 0
    not_padding_preds = max_preds[flattened_attention == 1]
    true_preds = not_padding_preds[reduce_ignored]

    print(true_preds.shape)
    result_text = ""
    next_upper = True
    for id, token in zip(true_preds, [token for tokens in all_tokens for token in tokens]):
        tag = id2tag[id]
        if next_upper:
            token = token.capitalize()
        punctuator, next_upper = tag_punctuator_map[tag]
        result_text += token + punctuator


    print(result_text)


def mark_ignored_tokens(offset_mapping):
    samples = []
    for sample_offset in offset_mapping:
        # create an empty array of -100
        sample_marks = np.ones(len(sample_offset), dtype=int) * -100
        arr_offset = np.array(sample_offset)

        # set labels whose first offset position is 0 and the second is not 0, only special tokens second is also 0
        sample_marks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = 0
        samples += sample_marks.tolist()

    return np.array(samples)


if __name__ == "__main__":
    register_logger()
    test_tokens = [['this', 'is', 'our', 'life', 'with', 'bees', 'and', 'this', 'is', 'our', 'life', 'without', 'bees', 'bees', 'are', 'the', 'most', 'important', 'pollinators', 'of', 'our', 'fruits', 'and', 'vegetables', 'and', 'flowers', 'and', 'crops', 'like', 'alfalfa', 'hay', 'that', 'feed', 'our', 'farm', 'animals', 'more', 'than', 'one', 'third', 'of', 'the', 'worlds', 'crop', 'production', 'is', 'dependent', 'on', 'bee', 'pollination', 'but', 'the', 'ironic', 'thing', 'is', 'that', 'bees', 'are', 'not', 'out', 'there', 'pollinating', 'our', 'food', 'intentionally', 'theyre', 'out', 'there', 'because', 'they', 'need', 'to', 'eat', 'bees', 'get', 'all', 'of', 'the', 'protein', 'they', 'need', 'in', 'their', 'diet', 'from', 'pollen', 'and', 'all', 'of', 'the', 'carbohydrates', 'they', 'need', 'from', 'nectar', 'theyre', 'flowerfeeders', 'and', 'as', 'they', 'move', 'from', 'flower', 'to', 'flower', 'basically', 'on', 'a', 'shopping', 'trip', 'at', 'the', 'local', 'floral', 'mart', 'they', 'end', 'up', 'providing', 'this']]
    test_tags = [['O', 'O', 'O', 'O', 'O', 'C', 'O', 'O', 'O', 'O', 'O', 'O', 'P', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'P', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'P', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'P', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'P', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'P', 'O', 'C', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'C', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'C', 'O', 'O', 'O', 'O', 'O']]
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    config = DistilBertConfig.from_pretrained("models/punctuator")
    model = DistilBertForTokenClassification.from_pretrained(
        "models/punctuator", config=config
    )
    model.eval()
    device = (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    model.to(device)

    inputs = tokenizer(
        test_tokens,
        is_split_into_words=True,
        padding=True,
        truncation=True,
        return_offsets_mapping=True,
        return_tensors="pt"
    )

    mraks = mark_ignored_tokens(inputs["offset_mapping"])

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    outputs = model(input_ids, attention_mask)

    process_results(device, outputs.logits, inputs["attention_mask"], mraks, test_tokens)


from transformers import (
    AdamW,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    BertConfig,
    BertForTokenClassification,
    BertTokenizerFast,
    get_constant_schedule_with_warmup,
)
from collections import namedtuple
from enum import Enum

ModelCollection = namedtuple('ModelCollection', ['config', 'tokenizer', 'model'])

class Models(Enum):
    DISTILBERT = ModelCollection(DistilBertConfig, DistilBertTokenizerFast, DistilBertForTokenClassification)
    BERT = ModelCollection(BertConfig, BertTokenizerFast, BertForTokenClassification)
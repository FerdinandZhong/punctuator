from collections import namedtuple
from enum import Enum

from transformers import (
    AutoConfig,
    AutoModel,
    BartConfig,
    BartModel,
    BartTokenizerFast,
    BertConfig,
    BertForTokenClassification,
    BertModel,
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaModel,
    RobertaTokenizerFast,
)

ModelCollection = namedtuple("ModelCollection", ["config", "tokenizer", "model"])


class Models(Enum):
    DISTILBERT = ModelCollection(
        DistilBertConfig, DistilBertTokenizerFast, DistilBertForTokenClassification
    )
    BERT_TOKEN_CLASSIFICATION = ModelCollection(
        BertConfig, BertTokenizerFast, BertForTokenClassification
    )
    BERT = ModelCollection(BertConfig, BertTokenizerFast, BertModel)
    BERT_PRETRAINING = ModelCollection(AutoConfig, BertTokenizerFast, AutoModel)
    ROBERTA = ModelCollection(RobertaConfig, RobertaTokenizerFast, RobertaModel)
    ROBERTA_TOKEN_CLASSIFICATION = ModelCollection(
        RobertaConfig, RobertaTokenizerFast, RobertaForTokenClassification
    )
    BART = ModelCollection(BartConfig, BartTokenizerFast, BartModel)

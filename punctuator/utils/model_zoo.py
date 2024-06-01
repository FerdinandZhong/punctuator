from collections import namedtuple
from enum import Enum
from punctuator.pykan.model import BertKanForTokenClassification

from transformers import (
    AutoConfig,
    AutoModel,
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

ModelCollection = namedtuple("ModelCollection", ["config", "tokenizer", "model", "backbone_model"])


class Models(Enum):
    DISTILBERT = ModelCollection(
        DistilBertConfig, DistilBertTokenizerFast, DistilBertForTokenClassification, DistilBertForTokenClassification
    )
    BERT_TOKEN_CLASSIFICATION = ModelCollection(
        BertConfig, BertTokenizerFast, BertForTokenClassification, BertForTokenClassification
    )
    BERT = ModelCollection(BertConfig, BertTokenizerFast, BertModel, BertModel)
    BERT_PRETRAINING = ModelCollection(AutoConfig, BertTokenizerFast, AutoModel, AutoModel)
    ROBERTA = ModelCollection(RobertaConfig, RobertaTokenizerFast, RobertaModel, RobertaModel)
    ROBERTA_TOKEN_CLASSIFICATION = ModelCollection(
        RobertaConfig, RobertaTokenizerFast, RobertaForTokenClassification, RobertaForTokenClassification
    )
    BERT_KAN = ModelCollection(
        BertConfig,
        BertTokenizerFast,
        BertKanForTokenClassification,
        BertModel
    )

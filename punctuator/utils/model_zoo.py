from collections import namedtuple
from enum import Enum

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

from punctuator.pykan.model import (
    BertKanForTokenClassification,
    BertKanForTokenClassification2,
)

ModelCollection = namedtuple(
    "ModelCollection", ["config", "tokenizer", "model", "backbone_model"]
)


def model_type(model_name):
    return Models[model_name.upper()]


class Models(Enum):
    DISTILBERT = ModelCollection(
        DistilBertConfig,
        DistilBertTokenizerFast,
        DistilBertForTokenClassification,
        DistilBertForTokenClassification,
    )
    BERT_TOKEN_CLASSIFICATION = ModelCollection(
        BertConfig,
        BertTokenizerFast,
        BertForTokenClassification,
        BertForTokenClassification,
    )
    BERT = ModelCollection(BertConfig, BertTokenizerFast, BertModel, BertModel)
    BERT_PRETRAINING = ModelCollection(
        AutoConfig, BertTokenizerFast, AutoModel, AutoModel
    )
    ROBERTA = ModelCollection(
        RobertaConfig, RobertaTokenizerFast, RobertaModel, RobertaModel
    )
    ROBERTA_TOKEN_CLASSIFICATION = ModelCollection(
        RobertaConfig,
        RobertaTokenizerFast,
        RobertaForTokenClassification,
        RobertaForTokenClassification,
    )
    BERT_KAN = ModelCollection(
        BertConfig, BertTokenizerFast, BertKanForTokenClassification, BertModel
    )

    BERT_KAN_2 = ModelCollection(
        BertConfig, BertTokenizerFast, BertKanForTokenClassification2, BertModel
    )

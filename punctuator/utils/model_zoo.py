from collections import namedtuple
from enum import Enum

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BertConfig,
    BertForTokenClassification,
    BertModel,
    BertTokenizerFast,
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizerFast,
    LlamaConfig,
    Qwen2Config,
    Qwen2TokenizerFast,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaModel,
    RobertaTokenizerFast,
)
from transformers.models.qwen2 import Qwen2ForCausalLM

from punctuator.focal_loss.bert_focal_loss import (
    BertFocalLossForTokenClassification,
    FocalLossForTokenClassificationStep2,
    RobertaFocalLossForTokenClassification,
    RobertaFocalLossForTokenClassificationStep2,
)
from punctuator.pykan.model import (
    BertKanForTokenClassification,
    BertKanForTokenClassification2,
    BertKanForTokenClassificationFocalLoss,
)
from punctuator.rotary_bert.model import (
    RotaryBertConfig,
    RotaryBertFocalLossForTokenClassification,
    RotaryBertForPreTraining,
    RotaryBertModel,
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
    BERT_FOCAL_LOSS = ModelCollection(
        BertConfig,
        BertTokenizerFast,
        BertFocalLossForTokenClassification,
        BertModel,
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

    BERT_KAN_FOCAL_LOSS = ModelCollection(
        BertConfig, BertTokenizerFast, BertKanForTokenClassificationFocalLoss, BertModel
    )

    ROFORMER_FOCAL_LOSS = ModelCollection(
        RotaryBertConfig,
        BertTokenizerFast,
        RotaryBertFocalLossForTokenClassification,
        BertModel,
    )

    ROFORMER_PRETRAIN = ModelCollection(
        RotaryBertConfig,
        BertTokenizerFast,
        RotaryBertForPreTraining,
        BertModel,
    )

    ROTARY_BERT_FOCAL_LOSS = ModelCollection(
        RotaryBertConfig,
        BertTokenizerFast,
        RotaryBertFocalLossForTokenClassification,
        RotaryBertModel,
    )

    QWEN2 = ModelCollection(
        Qwen2Config, Qwen2TokenizerFast, Qwen2ForCausalLM, Qwen2ForCausalLM
    )

    LLAMA31 = ModelCollection(
        LlamaConfig, AutoTokenizer, AutoModelForCausalLM, AutoModelForCausalLM
    )

    STEP2_FOCAL_LOSS = ModelCollection(
        BertConfig,
        BertTokenizerFast,
        FocalLossForTokenClassificationStep2,
        BertFocalLossForTokenClassification,
    )

    ROBERTA_FOCAL_LOSS = ModelCollection(
        RobertaConfig,
        RobertaTokenizerFast,
        RobertaFocalLossForTokenClassification,
        RobertaModel,
    )

    ROBERTA_STEP2_FOCAL_LOSS = ModelCollection(
        RobertaConfig,
        RobertaTokenizerFast,
        RobertaFocalLossForTokenClassificationStep2,
        RobertaFocalLossForTokenClassification,
    )

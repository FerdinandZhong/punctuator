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

logger = logging.getLogger(__name__)


class InferenceArguments(BaseModel):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Args:
        model_name_or_path(str): name or path of pre-trained model
        tokenizer_name(str): name of pretrained tokenizer
        tag2id_storage_path(str): tag2id storage path
    """
    model_name_or_path: str
    tokenizer_name: str
    tag2id_storage_path: str


# whole pipeline running in the seperate process, provide a function for user to call, use socket for communication
class InferencePipeline():
    """Pipeline for inference

    """
    def __init__(self, inference_arguments, socket_address, batch_size):
        self.arguments = inference_arguments
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            self.arguments.tokenizer_name
        )
        self.classifer = DistilBertForTokenClassification.from_pretrained(
            inference_arguments.model_name_or_path
        )


    def tokenize(self):
        pass


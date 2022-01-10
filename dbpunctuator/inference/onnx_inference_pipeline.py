import json
import logging

import torch
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from .inference_pipeline import InferencePipeline

logger = logging.getLogger(__name__)


class OnnxInferencePipeline(InferencePipeline):
    def __init__(self, inference_arguments, verbose=False):
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            inference_arguments.tokenizer_name
        )

        # TODO: initialize onnx model
        self.tag2punctuator = inference_arguments.tag2punctuator

        self._reset_values()
        self.verbose = verbose

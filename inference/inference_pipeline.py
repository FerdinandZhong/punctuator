import json
import logging
import struct
from itertools import filterfalse

import numpy as np
import signal
import torch
from pydantic import BaseModel
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from utils.constant import (
    DIGIT_MASK,
    LENGTH_BYTE_FORMAT,
    LENGTH_BYTE_LENGTH,
    NUM_BYTE_FORMAT,
    NUM_BYTE_LENGTH,
    TAG_PUNCTUATOR_MAP,
)
from utils.utils import recv_all, register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


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
class InferencePipeline:
    """Pipeline for inference"""

    def __init__(self, inference_arguments):
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
        with open(inference_arguments.tag2id_storage_path, "r") as fp:
            tag2id = json.load(fp)
            self.id2tag = {id: tag for tag, id in tag2id.items()}

        self.digit_indexes = []
        self.all_tokens = []
        self.outputs = []

    def pre_process(self, inputs):
        for input in inputs:
            input_tokens = input.split()
            digits = dict(
                list(filterfalse(lambda x: not x[1].isdigit(), enumerate(input_tokens)))
            )
            for index_key in digits.keys():
                input_tokens[index_key] = DIGIT_MASK
            self.digit_indexes.append(digits)
            self.all_tokens.append(input_tokens)
        return self

    def tokenize(self):
        self.inputs = self.tokenizer(
            self.all_tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        self.marks = self._mark_ignored_tokens(self.inputs["offset_mapping"])
        return self

    def classify(self):
        input_ids = self.inputs["input_ids"].to(self.device)
        attention_mask = self.inputs["attention_mask"].to(self.device)
        self.logits = self.classifer(input_ids, attention_mask).logits
        return self

    def post_process(self):
        if self.device.type == "cuda":
            max_preds = self.logits.argmax(dim=2).detach().cpu().numpy()
        else:
            max_preds = self.logits.argmax(dim=2).detach().numpy()

        reduce_ignored_marks = self.marks >= 0

        next_upper = True
        for pred, reduce_ignored, tokens, digit_index in zip(
            max_preds, reduce_ignored_marks, self.all_tokens, self.digit_indexes
        ):
            true_pred = pred[reduce_ignored]
            result_text = ""
            for id, (index, token) in zip(true_pred, enumerate(tokens)):
                tag = self.id2tag[id]
                if index in digit_index:
                    token = digit_index[index]
                if next_upper:
                    token = token.capitalize()
                punctuator, next_upper = TAG_PUNCTUATOR_MAP[tag]
                result_text += token + punctuator
            self.outputs.append(result_text.strip())
        return self.outputs

    def punctuation(self, inputs):
        return self.pre_process(inputs).tokenize().classify().post_process()

    def _mark_ignored_tokens(self, offset_mapping):
        samples = []
        for sample_offset in offset_mapping:
            # create an empty array of -100
            sample_marks = np.ones(len(sample_offset), dtype=int) * -100
            arr_offset = np.array(sample_offset)

            # set labels whose first offset position is 0 and the second is not 0, only special tokens second is also 0
            sample_marks[(arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)] = 0
            samples.append(sample_marks.tolist())

        return np.array(samples)


class InferenceServer:
    """Inference server"""

    def __init__(self, inference_args, conn, termination, check_interval=0.1) -> None:
        """Server for receiving tasks from client and do punctuation

        Args:
            server_address (str, optional): [server socket address]. Defaults to "/tmp/socket".
        """
        self.inference_pipeline = InferencePipeline(inference_args)
        self.conn = conn
        self.termination = termination
        self.check_interval = check_interval

    # data structure: |num|length|text|length|text...
    def punctuation(self):
        try:
            inputs = self.conn.recv()
            outputs = self.inference_pipeline.punctuation(inputs)
            self.conn.send(outputs)
        except OSError as err:
            logger.warning(f"error receiving inputs: {err}")
        except struct.error as err:
            logger.warning(f"struct unpack error: {err}")

    def run(self):
        assert self.inference_pipeline, "no inference pipeline set up"
        logger.info("server is running")
        while True:
            try:
                if self.termination.is_set():
                    logger.info("termination is set")
                    break
                if self.conn.poll(self.check_interval):
                    self.punctuation()
            except (struct.error, OSError) as err:
                logger.warning(f"struct unpack error: {err}")
                raise err
            except KeyboardInterrupt:
                logger.warning("punctuator shut down by keyboard interrupt")
                break
        self.terminate()

    def terminate(self):
        logger.info("terminate punctuation server")

        self.conn.close()

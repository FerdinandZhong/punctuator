import json
import logging
import struct
from functools import wraps
from itertools import filterfalse

import numpy as np
import torch
from pydantic import BaseModel
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from utils.constant import DIGIT_MASK, TAG_PUNCTUATOR_MAP
from utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def verbose(attr_to_log):
    def wrapper_out(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            self = func(self, *args, **kwargs)
            if self.verbose:
                logger.debug(
                    f'After {func.__name__}, {attr_to_log} is generated as "{getattr(self, attr_to_log)}"'
                )
            return self

        return wrapper

    return wrapper_out


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

    def __init__(self, inference_arguments, verbose=False):
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

        self._reset_values()
        self.verbose = verbose

    @verbose("all_tokens")
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

    @verbose("tokenized_input_ids")
    def tokenize(self):
        tokenized_inputs = self.tokenizer(
            self.all_tokens,
            is_split_into_words=True,
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        self.marks = self._mark_ignored_tokens(tokenized_inputs["offset_mapping"])
        self.tokenized_input_ids = tokenized_inputs["input_ids"].to(self.device)
        self.attention_mask = tokenized_inputs["attention_mask"].to(self.device)
        return self

    @verbose("argmax_preds")
    def classify(self):
        logits = self.classifer(self.tokenized_input_ids, self.attention_mask).logits
        if self.device.type == "cuda":
            self.argmax_preds = logits.argmax(dim=2).detach().cpu().numpy()
        else:
            self.argmax_preds = logits.argmax(dim=2).detach().numpy()
        return self

    @verbose("outputs")
    def post_process(self):
        reduce_ignored_marks = self.marks >= 0

        for pred, reduce_ignored, tokens, digit_index in zip(
            self.argmax_preds, reduce_ignored_marks, self.all_tokens, self.digit_indexes
        ):
            next_upper = True
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

        return self

    def punctuation(self, inputs):
        self._reset_values()
        return self.pre_process(inputs).tokenize().classify().post_process().outputs

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

    def _reset_values(self):
        self.digit_indexes = []
        self.all_tokens = []
        self.outputs = []


class InferenceServer:
    """Inference server"""

    def __init__(
        self, inference_args, conn, termination, check_interval, verbose=False
    ) -> None:
        """Server for receiving tasks from client and do punctuation

        Args:
            server_address (str, optional): [server socket address]. Defaults to "/tmp/socket".
        """
        self.inference_pipeline = InferencePipeline(inference_args, verbose=verbose)
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
                if (
                    self.conn.poll(self.check_interval)
                    and not self.termination.is_set()
                ):
                    self.punctuation()
                if self.termination.is_set():
                    logger.info("termination is set")
                    break
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

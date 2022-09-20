import json
import logging
import re
from functools import wraps
from os import environ
from typing import Dict, Optional

import numpy as np
import torch
from plane.pattern import EMAIL, TELEPHONE
from pydantic import BaseModel
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from dbpunctuator.utils import (
    CURRENCY,
    CURRENCY_TOKEN,
    EMAIL_TOKEN,
    NUMBER,
    NUMBER_TOKEN,
    TELEPHONE_TOKEN,
    URL,
    URL_TOKEN,
    chinese_split,
    is_ascii,
)

logger = logging.getLogger(__name__)

num_regex = re.compile(f"{NUMBER.pattern}")
tel_regex = re.compile(f"{TELEPHONE.pattern}")
currency_regex = re.compile(f"{CURRENCY.pattern}")
email_regex = re.compile(f"{EMAIL.pattern}")
url_regex = re.compile(f"{URL.pattern}")


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
        tag2punctuator(Dict[str, tuple]): tag to punctuator mapping.
            dbpunctuator.utils provides two mappings for English and Chinese
                NORMAL_TOKEN_TAG = "O"
                DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP = {
                    NORMAL_TOKEN_TAG: ("", False),
                    "COMMA": (",", False),
                    "PERIOD": (".", True),
                    "QUESTIONMARK": ("?", True),
                    "EXLAMATIONMARK": ("!", True),
                }

                DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP = {
                    NORMAL_TOKEN_TAG: ("", False),
                    "C_COMMA": ("，", False),
                    "C_PERIOD": ("。", True),
                    "C_QUESTIONMARK": ("? ", True),
                    "C_EXLAMATIONMARK": ("! ", True),
                    "C_DUNHAO": ("、", False),
                }
            for own fine-tuned model with different tags, pass in your own mapping
        tag2id_storage_path(Optional[str]): tag2id storage path. Default one is from model config. Pass in this argument if your model doesn't have a tag2id inside config # noqa: E501
        gpu_device(int): specific gpu card index, default is the CUDA_VISIBLE_DEVICES from environ
    """

    model_name_or_path: str
    tokenizer_name: str
    tag2punctuator: Dict[str, tuple]
    tag2id_storage_path: Optional[str]
    gpu_device: int = environ.get("CUDA_VISIBLE_DEVICES", 0)


# whole pipeline running in the seperate process, provide a function for user to call, use socket for communication
class InferencePipeline:
    """Pipeline for inference"""

    def __init__(self, inference_arguments, verbose=False):
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{inference_arguments.gpu_device}")
            logger.info(f"device type: {self.device.type}")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            inference_arguments.tokenizer_name
        )
        self.classifier = DistilBertForTokenClassification.from_pretrained(
            inference_arguments.model_name_or_path
        ).to(self.device)
        if inference_arguments.tag2id_storage_path:
            with open(inference_arguments.tag2id_storage_path, "r") as fp:
                tag2id = json.load(fp)
                self.id2tag = {id: tag for tag, id in tag2id.items()}
        else:
            self.id2tag = self.classifier.config.id2label
        self.tag2punctuator = inference_arguments.tag2punctuator
        self.max_sequence_length = (
            self.classifier.config.max_position_embeddings // 2
        )  # for padding and subword

        self._reset_values()
        self.verbose = verbose

    @verbose("all_tokens")
    def pre_process(self, inputs):
        def _input_process(input_tokens):

            special_token_index = {}
            for index, token in enumerate(input_tokens):
                if email_regex.match(token):
                    input_tokens[index] = EMAIL_TOKEN
                    special_token_index[index] = token
                    continue
                if url_regex.match(token):
                    input_tokens[index] = URL_TOKEN
                    special_token_index[index] = token
                    continue
                if currency_regex.match(token):
                    input_tokens[index] = CURRENCY_TOKEN
                    special_token_index[index] = token
                    continue
                if tel_regex.match(token):
                    input_tokens[index] = TELEPHONE_TOKEN
                    special_token_index[index] = token
                    continue
                if num_regex.match(token):
                    input_tokens[index] = NUMBER_TOKEN
                    special_token_index[index] = token
                    continue
            return input_tokens, special_token_index

        index = 0
        last_is_split = False
        self.split_inputs_indexes = []
        for input in inputs:
            input_tokens = chinese_split(input).split()
            while len(input_tokens) > self.max_sequence_length:
                processed_input_tokens, special_token_index = list(
                    _input_process(input_tokens[: self.max_sequence_length])
                )
                self.special_token_indexes.append(special_token_index)
                self.all_tokens.append(processed_input_tokens)
                self.split_inputs_indexes.append(index)
                input_tokens = input_tokens[self.max_sequence_length :]
                index += 1
                last_is_split = True
            else:
                if last_is_split:
                    self.split_inputs_indexes.append(index)
                    last_is_split = False
                index += 1
                processed_input_tokens, special_token_index = list(
                    _input_process(input_tokens)
                )
                self.special_token_indexes.append(special_token_index)
                self.all_tokens.append(processed_input_tokens)
        logger.info(f"self split indexes: {self.split_inputs_indexes}")
        return self

    @verbose("tokenized_input_ids")
    def tokenize(self):
        tokenized_inputs = self.tokenizer(
            self.all_tokens,
            is_split_into_words=True,
            padding=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        self.marks = self._mark_ignored_tokens(tokenized_inputs["offset_mapping"])
        self.tokenized_input_ids = tokenized_inputs["input_ids"].to(self.device)
        self.attention_mask = tokenized_inputs["attention_mask"].to(self.device)
        return self

    @verbose("argmax_preds")
    def classify(self):
        try:
            logits = self.classifier(
                self.tokenized_input_ids, self.attention_mask
            ).logits
            if self.device.type == "cuda":
                self.argmax_preds = logits.argmax(dim=2).detach().cpu().numpy()
            else:
                self.argmax_preds = logits.argmax(dim=2).detach().numpy()
        except RuntimeError as e:
            logger.error(f"error doing punctuation: {str(e)}")
        return self

    @verbose("outputs")
    def post_process(self):
        reduce_ignored_marks = self.marks >= 0

        self.outputs_labels = []
        temp_ouputs = ""
        temp_outputs_labels = []
        for input_index, (
            pred,
            reduce_ignored,
            tokens,
            special_token_index,
        ) in enumerate(
            zip(
                self.argmax_preds,
                reduce_ignored_marks,
                self.all_tokens,
                self.special_token_indexes,
            )
        ):
            next_upper = True
            true_pred = pred[reduce_ignored]

            result_text = ""
            output_labels = []
            for id, (index, token) in zip(true_pred, enumerate(tokens)):
                tag = self.id2tag[id]
                output_labels.append(tag)
                if index in special_token_index:
                    token = special_token_index[index]
                if next_upper:
                    token = token.capitalize()
                punctuator, next_upper = self.tag2punctuator[tag]
                if is_ascii(token):
                    result_text += token + punctuator + " "
                else:
                    result_text += token + punctuator
            if input_index in self.split_inputs_indexes:
                temp_ouputs += result_text.strip()
                temp_outputs_labels.extend(output_labels)
            else:
                if temp_ouputs and temp_outputs_labels:
                    self.outputs.append(temp_ouputs.strip())
                    self.outputs_labels.append(temp_outputs_labels)
                    temp_ouputs = ""
                    temp_outputs_labels = []

                self.outputs.append(result_text.strip())
                self.outputs_labels.append(output_labels)

        if temp_ouputs and temp_outputs_labels:
            self.outputs.append(temp_ouputs.strip())
            self.outputs_labels.append(temp_outputs_labels)

        return self

    def punctuation(self, inputs):
        self._reset_values()
        self.pre_process(inputs).tokenize().classify().post_process()

        return self.outputs, self.outputs_labels

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
        self.special_token_indexes = []
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

    def punctuation(self):
        try:
            inputs = self.conn.recv()
            outputs_tuple = self.inference_pipeline.punctuation(inputs)
            self.conn.send(outputs_tuple)
        except OSError as err:
            logger.warning(f"error receiving inputs: {err}")

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
            except OSError as err:
                logger.warning(f"sending output error: {err}")
                raise err
            except KeyboardInterrupt:
                logger.warning("punctuator shut down by keyboard interrupt")
                break
        self.terminate()

    def terminate(self):
        logger.info("terminate punctuation server")

        self.conn.close()

import json
import logging
import os
import selectors
import signal
import socket
import struct
from itertools import filterfalse
from threading import Event

import numpy as np
import torch
from pydantic import BaseModel
from transformers import DistilBertForTokenClassification, DistilBertTokenizerFast

from utils.constant import DIGIT_MASK, TAG_PUNCTUATOR_MAP
from utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


# byte format
NUM_BYTE_FORMAT = "!H"
LENGTH_BYTE_FORMAT = "!I"

NUM_BYTE_LENGTH = 2
LENGTH_BYTE_LENGTH = 4


class InferenceArguments(BaseModel):
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.

    Args:
        model_name_or_path(str): name or path of pre-trained model
        tokenizer_name(str): name of pretrained tokenizer
        tag2id_storage_path(str): tag2id storage path
    """

    model_name_or_path: str
    tokenizer_name: str
    id2tag_storage_path: str


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
        with open(inference_arguments.id2tag_storage_path, "r") as fp:
            self.id2tag = json.load(fp)

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
            self.input_tokens,
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
        self.logits = self.model(input_ids, attention_mask).logits
        return self

    def post_process(self):
        if self.device.type == "cuda":
            max_preds = self.logits.argmax(dim=2).detach().cpu().numpy()
        else:
            max_preds = self.logits.argmax(dim=2).detach().numpy()

        reduce_ignored_marks = self.marks >= 0

        next_upper = True
        for pred, reduce_ignored, tokens, digit_index in zip(
            max_preds, reduce_ignored_marks, self.input_tokens, self.digit_indexes
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
            self.outputs.append(result_text)
        return self.outputs

    def punctuation(self, inputs):
        return self.pre_process(inputs).tokenize().classify().post_process()

    def _mark_ignored_tokens(offset_mapping):
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

    def __init__(
        self,
        inference_args,
        socket_address="/tmp/socket",
    ) -> None:
        """Server for receiving tasks from client and do punctuation

        Args:
            server_address (str, optional): [server socket address]. Defaults to "/tmp/socket".
        """
        try:
            os.remove(socket_address)
        except OSError:
            pass
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.bind(socket_address)
        self.socket.listen()
        logger.info(f"inference server is listening on {socket_address}")
        self.socket.setblocking(False)

        self.sel = selectors.DefaultSelector()
        self.sel.register(self.socket, selectors.EVENT_READ, data=self.accept)

        self.inference_pipeline = InferencePipeline(inference_args)

    def accept(self, sock):
        conn, addr = sock.accept()  # Should be ready to read
        logger.info(f"accepted connection from {addr}")
        conn.setblocking(False)
        events = selectors.EVENT_READ | selectors.EVENT_WRITE
        self.sel.register(conn, events, data=self.punctuation)
        if self.termination.is_set():
            self.sel.unregister(sock)

    # data structure: |num|length|text|length|text...
    def punctuation(self, conn):
        num_bytes = conn.recv(NUM_BYTE_LENGTH)
        if num_bytes:
            all_payloads = []
            output_array = bytearray()
            output_array += num_bytes
            num = struct.unpack(NUM_BYTE_FORMAT, num_bytes)[0]
            while num > 0:
                length_bytes = conn.recv(LENGTH_BYTE_LENGTH)
                length = struct.unpack(LENGTH_BYTE_FORMAT, length_bytes)[0]
                all_payloads.append(self._recv_all(conn, length).decode())
                num -= 1
            outputs = self.inference_pipeline.punctuation()
            for output in outputs:
                output_array += (
                    struct.pack(LENGTH_BYTE_FORMAT, len(output.encode()))
                    + output.encode()
                )
            self.conn.send(output_array)

    def _recv_all(self, conn, length):
        buffer = bytearray(length)
        mv = memoryview(buffer)
        size = 0
        while size < length:
            packet = conn.recv_into(mv)
            mv = mv[packet:]
            size += packet
        return buffer

    def _init_termination(self):
        """init signal handler and termination event"""
        self.termination = Event()
        signal.signal(signal.SIGTERM, self._terminate)
        signal.signal(signal.SIGINT, self._terminate)

    def _terminate(self, signum, frame):
        """graceful shutdown everything"""
        logger.info(f"[{signum}] terminate server: {frame}")

        self.termination.set()

    def run(self):
        assert self.inference_pipeline, "no inference pipeline set up"
        logger.info("server is running")
        while True:
            events = self.sel.select(timeout=None)
            for key, _ in events:
                callback = key.data
                callback(key.fileobj)
            if self.termination.is_set():
                logger.info("termination is set")
                break
        self.socket.close()
        self.sel.close()


class InferenceClient:
    """Inference client"""


# class InferenceInterface:
#     """Interface for using the inference"""

#     @classmethod
#     def launch_inference(cls):
#         # TODO: launch pipeline instance in
#         pass

#     @classmethod
#     def pass_text(cls):
#         pass

# TODO: sdk interface
# launch_service()
# punctuatation()

# model inference --> processes ---> async + dserving + batching
# unix socket to call

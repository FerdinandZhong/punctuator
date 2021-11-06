import logging
import multiprocessing as mp
import signal
import struct
import threading
from threading import Thread
from time import sleep
from typing import List

from inference.inference_pipeline import InferenceServer
from utils.constant import (
    LENGTH_BYTE_FORMAT,
    LENGTH_BYTE_LENGTH,
    NUM_BYTE_FORMAT,
    NUM_BYTE_LENGTH,
)
from utils.utils import recv_all, register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


class InferenceClient:
    """Inference client"""

    def __init__(self, conn, check_interval=0.1) -> None:
        self.conn = conn
        self.check_interval = check_interval

    def punctuation(self, inputs):
        self.conn.send(inputs)
        while True:
            try:
                if self.conn.poll(self.check_interval):
                    outputs = self.conn.recv()
                    return outputs
            except (struct.error, OSError) as err:
                logger.warning(f"{self.name} struct unpack error: {err}")
                raise err

    def terminate(self):
        """graceful shutdown everything"""
        logger.info("terminate the client")

        self.conn.close()


class Inference:
    """Interface for using the punctuator"""

    def __init__(
        self,
        inference_args,
        socket_address="/tmp/punctuator.socket",
        method="spawn",
        check_interval=0.1,
    ) -> None:
        self.termination = mp.get_context(method).Event()
        self.method = method
        self.inference_args = inference_args
        self.socket_address = socket_address

        self._init_termination()
        self._produce_server()
        self.thread = Thread(target=self._run, args=(check_interval,))
        self.thread.start()

    def _produce_server(self):
        logger.info("set up punctuator")
        self.c_conn, self.s_conn = mp.Pipe(True)
        server = InferenceServer(
            inference_args=self.inference_args,
            conn=self.s_conn,
            termination=self.termination,
        )
        self.server_process = mp.get_context(self.method).Process(
            target=server.run,
        )

        logger.info("start running punctuator")
        self.server_process.start()

        logger.info("start client")
        self.client = InferenceClient(conn=self.c_conn)


    def _init_termination(self):
        """init signal handler and termination event"""
        self.shutdown = threading.Event()
        signal.signal(signal.SIGTERM, self._terminate)
        signal.signal(signal.SIGINT, self._terminate)

    def _terminate(self, signum, frame):
        """graceful shutdown everything"""
        logger.info(f"[{signum}] terminate inference: {frame}")

        self.shutdown.set()
        self.termination.set()
        self.client.terminate()

    def _run(self, check_interval):
        while not self.shutdown.is_set():
            if not self.server_process.exitcode is None:
                logger.warning("punctuator is no longer working, restart")
                self._produce_server()
            sleep(check_interval)
        logger.info("terminate the punctuator")
        # self.server_process.terminate()

    def punctuation(self, inputs: List[str]):
        try:
            outputs = self.client.punctuation(inputs)
            return outputs
        except Exception as err:
            logger.error(f"error doing punctuation with details {str(err)}")
        return None


if __name__ == "__main__":
    from inference.inference_pipeline import InferenceArguments

    args = InferenceArguments(
        model_name_or_path="models/punctuator",
        tokenizer_name="distilbert-base-uncased",
        tag2id_storage_path="models/tag2id.json",
    )

    inference = Inference(inference_args=args)

    test_texts = [
        "how are you its been ten years since we met in shanghai im really happen to meet you again whats your current phone number",
        "my number is 82732212",
    ]
    logger.info(f"testing result {inference.punctuation(test_texts)}")

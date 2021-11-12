import logging
import multiprocessing as mp
import signal
import struct
import threading
from threading import Thread
from time import sleep
from typing import List

from inference.inference_pipeline import InferenceServer
from utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


class InferenceClient:
    """Inference client for communicating with punctuator server"""

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
        method="spawn",
        server_check_interval=0.1,
        task_check_interval=0.05,
        verbose=False,
    ) -> None:
        """Inference class for using the punctuator

        Args:
            inference_args (InferenceArguments): inference arguments
            method (str, optional): "fork" or "spawn". Defaults to "spawn".
            server_check_interval (float, optional): interval to check punctuator running status. Defaults to 0.1.
            task_check_interval (float, optional): interval to check new task. Defaults to 0.05.
            verbose (bool, optional): whether to ouput punctuation progress. Defaults to False.
        """
        self.termination = mp.get_context(method).Event()
        self.method = method
        self.inference_args = inference_args
        self.verbose = verbose

        self._init_termination()
        self._produce_server(task_check_interval)
        self.thread = Thread(target=self._run, args=(server_check_interval,))
        self.thread.start()

    def _produce_server(self, task_check_interval):
        logger.info("set up punctuator")
        self.c_conn, self.s_conn = mp.Pipe(True)
        server = InferenceServer(
            inference_args=self.inference_args,
            conn=self.s_conn,
            termination=self.termination,
            check_interval=task_check_interval,
            verbose=self.verbose,
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
            if self.server_process.exitcode is not None:
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

    def terminate(self):
        self.shutdown.set()
        self.termination.set()
        self.client.terminate()

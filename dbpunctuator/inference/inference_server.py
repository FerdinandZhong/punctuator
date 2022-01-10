import logging

from .inference_pipeline import InferencePipeline

logger = logging.getLogger(__name__)


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

import logging

from .utils.utils import register_logger

# setup library logging
logger = logging.getLogger(__name__)
register_logger(logger)

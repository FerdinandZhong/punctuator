import logging
from functools import wraps

logger = logging.getLogger(__name__)


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

import logging
import re

from plane import ASCII_WORD

logger = logging.getLogger(__name__)


def remove_brackets_text(input):
    """remove brackets and text inside

    Args:
        input (string): text to apply the regex func to

    """
    return re.sub(r"\([^()]*\)", " ", input)


def keep_only_latin_characters(input):
    """keep only latin characters

    Args:
        input (string): text to apply the regex func to

    """
    regex = re.compile("[^\u0020-\u024F]")
    return regex.sub("", input)


def chinese_split(input):
    """Split Chinese input by:
    - Adding space between every Chinese character. Note: English word will remain as original

    Args:
        input (string): text to apply the regex func to
    """

    regex = re.compile(
        "(?P<%s>%s)" % (ASCII_WORD.name, ASCII_WORD.pattern), ASCII_WORD.flag
    )
    result = ""
    start = 0
    try:
        for t in regex.finditer(input):
            result += " ".join(
                [char for char in list(input[start : t.start()]) if char != " "]
            )
            result += " " + input[t.start() : t.end()] + " "
            start = t.end()
        result += " ".join([char for char in list(input[start:]) if char != " "])
    except TypeError:
        # mal row
        logger.warning(f"error parsing data: {input}")
    return result

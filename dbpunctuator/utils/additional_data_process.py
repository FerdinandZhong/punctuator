import logging
import re

from plane import CJK

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

    regex = re.compile("(?P<%s>%s)" % (CJK.name, CJK.pattern), CJK.flag)
    result = ""
    start = 0
    try:
        for t in regex.finditer(input):
            result += input[start : t.start()].strip()
            result += (
                " "
                + " ".join(
                    [char for char in list(input[t.start() : t.end()]) if char != " "]
                )
                + " "
            )
            start = t.end()
        result += input[start:].strip()
    except TypeError as err:
        # mal row
        logger.warning(f"parsing data: {input} with error: {str(err)}")
    return result

import sys
import unicodedata

from plane import build_new_regex

NORMAL_TOKEN_TAG = "O"
EMAIL_TOKEN = "<EMAIL>"
URL_TOKEN = "<URL>"
TELEPHONE_TOKEN = "<TEL>"
CURRENCY_TOKEN = "<CURRENCY>"
NUMBER_TOKEN = "<NUM>"
URL = build_new_regex(
    "url_checking",
    r"https?:\/\/[!-~]+|[!-~]+\.[-_a-z/]+",
)


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


DEFAULT_ENGLISH_NER_MAPPING = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QUESTIONMARK",
    "!": "EXLAMATIONMARK",
}

DEFAULT_CHINESE_NER_MAPPING = {
    "，": "C_COMMA",
    "。": "C_PERIOD",
    "？": "C_QUESTIONMARK",
    "！": "C_EXLAMATIONMARK",
    "、": "C_DUNHAO",
}


ALL_PUNCS = [
    c
    for c in range(sys.maxunicode)
    if unicodedata.category(chr(c)).startswith(("P", "Cc"))
]

currency_list = "|".join(
    [
        chr(c)
        for c in range(sys.maxunicode)
        if unicodedata.category(chr(c)).startswith(("Sc"))
    ]
)
CURRENCY = build_new_regex(
    "currency", r"(\{})\d+([.,]?\d*)*([A-Za-z]+)?".format(currency_list)
)

NUMBER = build_new_regex("number", r"[0-9]*[.]?[0-9]+[%]?")
# byte format
NUM_BYTE_FORMAT = "!H"
LENGTH_BYTE_FORMAT = "!I"

NUM_BYTE_LENGTH = 2
LENGTH_BYTE_LENGTH = 4

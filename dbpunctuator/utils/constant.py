import sys
import unicodedata

NORMAL_TOKEN_TAG = "O"
DIGIT_MASK = "<NUM>"

DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP = {
    NORMAL_TOKEN_TAG: ("", False),
    "C": (",", False),
    "P": (".", True),
    "Q": ("?", True),
    "E": ("!", True),
    "CO": (":", False),
}
DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP = {
    NORMAL_TOKEN_TAG: ("", False),
    "CC": ("，", False),
    "CP": ("。", True),
    "CQ": ("? ", True),
    "CE": ("! ", True),
    "CL": ("：", True),
    "CD": ("、", False),
    "P": (".", True),
}


DEFAULT_ENGLISH_NER_MAPPING = {
    ",": "C",
    ".": "P",
    "?": "Q",
    "!": "E",
    ":": "CO",
}
DEFAULT_CHINESE_NER_MAPPING = {
    ".": "P",  # for latin specific situation in Chinese text
    "，": "CC",
    "。": "CP",
    "？": "CQ",
    "！": "CE",
    "：": "CL",
    "、": "CD",
}


ALL_PUNCS = [
    c
    for c in range(sys.maxunicode)
    if unicodedata.category(chr(c)).startswith(("P", "Cc"))
]


# byte format
NUM_BYTE_FORMAT = "!H"
LENGTH_BYTE_FORMAT = "!I"

NUM_BYTE_LENGTH = 2
LENGTH_BYTE_LENGTH = 4

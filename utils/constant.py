TAG_PUNCTUATOR_MAP = {
    "O": (" ", False),
    "C": (", ", False),
    "P": (". ", True),
    "Q": ("? ", True),
    "E": ("! ", True),
}

DEFAULT_NER_MAPPING = {",": "C", ".": "P", "?": "Q", "!": "E"}
DIGIT_MASK = "<num>"


# byte format
NUM_BYTE_FORMAT = "!H"
LENGTH_BYTE_FORMAT = "!I"

NUM_BYTE_LENGTH = 2
LENGTH_BYTE_LENGTH = 4

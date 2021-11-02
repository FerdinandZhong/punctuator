TAG_PUNCTUATOR_MAP = {
    "O": (" ", False),
    "C": (", ", False),
    "P": (". ", True),
    "Q": ("? ", True),
    "E": ("! ", True)
}

NER_MAPPING = {",": "C", ".": "P", "?": "Q", "!": "E"}
DIGIT_MASK = "<num>"
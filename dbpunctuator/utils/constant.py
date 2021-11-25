DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP = {
    "O": (" ", False),
    "C": (", ", False),
    "P": (". ", True),
    "Q": ("? ", True),
    "E": ("! ", True),
}
DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP = {
    "O": ("", False),
    "CC": ("，", False),
    "CP": ("。", True),
    "CQ": ("? ", True),
    "CE": ("! ", True),
    "CO": ("：", True),
    "CD": ("、", False),
}


DEFAULT_ENGLISH_NER_MAPPING = {",": "C", ".": "P", "?": "Q", "!": "E"}
DEFAULT_CHINESE_NER_MAPPING = {
    "，": "CC",
    "。": "CP",
    "？": "CQ",
    "！": "CE",
    "：": "CO",
    "、": "CD",
}

CHINESE_PUNCS = (
    "！？｡。＂＃＄％＆＇（）＊＋，－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠《》｢｣､、〃》「」『』【】〔〕〖〗〘〙〚〛〜〝〞〟〰〾〿–—‘’‛“”„‟…‧﹏."
)

DIGIT_MASK = "<num>"


# byte format
NUM_BYTE_FORMAT = "!H"
LENGTH_BYTE_FORMAT = "!I"

NUM_BYTE_LENGTH = 2
LENGTH_BYTE_LENGTH = 4

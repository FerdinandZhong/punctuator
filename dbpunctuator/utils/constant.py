TAG_PUNCTUATOR_MAP = {
    "O": (" ", False),
    "C": (", ", False),
    "P": (". ", True),
    "Q": ("? ", True),
    "E": ("! ", True),
}
# TODO: optimize this part

DEFAULT_ENGLISH_NER_MAPPING = {",": "C", ".": "P", "?": "Q", "!": "E", "'": "a"}
DEFAULT_CHINESE_NER_MAPPING = {
    "，": "CC",
    "。": "CP",
    "?": "Q",
    "!": "E",
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

DEFAULT_TAG_ID = {"E": 0, "O": 1, "P": 2, "C": 3, "Q": 4}

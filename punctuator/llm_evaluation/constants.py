from punctuator.utils import NORMAL_TOKEN_TAG

LLM_CHAT_MESSAGES_BERT_OUTPUT = [
    {
        "role": "system",
        "content": "In the following text, '{token}' refers to the poistion of punctuation. Replace the '{token}' with correct punctuations.",
    },
    {
        "role": "user",
        "content": "it involves developing skills such as listening{token} speaking{token} pronunciation{token} vocabulary{token} grammar{token} and discourse{token}",
    },
    {
        "role": "assistant",
        "content": "It involves developing skills such as listening, speaking, pronunciation, vocabulary, grammar, and discourse.",
    },
    {
        "role": "user",
        "content": "东方主义{token}的影响非常广泛{token}它不仅影响了西方对东方世界的看法{token}还对艺术{token}文学{token}音乐电影等多个领域产生了重要影响{token}",
    },
    {
        "role": "assistant",
        "content": "东方主义的影响非常广泛，它不仅影响了西方对东方世界的看法还对艺术，文学，音乐电影等多个领域产生了重要影响。",
    },
]

LLM_CHAT_MESSAGES_RAW = [
    {
        "role": "system",
        "content": "Restore the punctuations to the following text without any extra explanation. ",
    },
    {
        "role": "user",
        "content": "it involves developing skills such as listening speaking pronunciation vocabulary grammar and discourse",
    },
    {
        "role": "assistant",
        "content": "It involves developing skills such as listening, speaking, pronunciation, vocabulary, grammar, and discourse.",
    },
    {"role": "user", "content": "东方主义的影响非常广泛它不仅影响了西方对东方世界的看法还对艺术文学音乐电影等多个领域产生了重要影响"},
    {
        "role": "assistant",
        "content": "东方主义的影响非常广泛，它不仅影响了西方对东方世界的看法还对艺术，文学，音乐电影等多个领域产生了重要影响。",
    },
]

PUNCT_SPECIAL_TOKEN = "#"

CROSS_LANG_PUNCT_MAPPINGS = {
    "？": "?",
    "！": "!",
    "。": ".",
    "、": ",",
    "，": ",",
    ": “": ",",
    "（": "(",
    "）": ")",
}

LABEL2PUNCT = {
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTIONMARK": "?",
    "EXCLAMATIONMARK": "!",
    "PUNCT": PUNCT_SPECIAL_TOKEN,
}

PUNCT2LABEL = {
    ",": "COMMA",
    ".": "PERIOD",
    "?": "QUESTIONMARK",
    "!": "EXCLAMATIONMARK",
    PUNCT_SPECIAL_TOKEN: NORMAL_TOKEN_TAG,
}

LABEL2ID = {
    NORMAL_TOKEN_TAG: 0,
    "COMMA": 1,
    "PERIOD": 2,
    "QUESTIONMARK": 3,
    "EXCLAMATIONMARK": 4,
    "PUNCT": -1,
}

from punctuator.utils import NORMAL_TOKEN_TAG

LLM_CHAT_MESSAGES_BERT_OUTPUT = [
    {
        "role": "system",
        "content": "In the following text, '{token}' refers to the position of punctuation. Replace the '{token}' with correct punctuation.",
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

LLM_CHAT_MESSAGES_BERT_OUTPUT_5_EN_SAMPLE = [
    {
        "role": "system",
        "content": "In the following text, '{token}' refers to the poistion of punctuation. Replace the '{token}' with correct punctuations.",
    },
    {
        "role": "user",
        "content": "most of the time{token} art and science stare at each other across a gulf of mutual incomprehension{token} there is great confusion",
    },
    {
        "role": "assistant",
        "content": "Most of the time, art and science stare at each other across a gulf of mutual incomprehension. There is great confusion",
    },
    {
        "role": "user",
        "content": "good morning{token} i think{token} as a grumpy eastern european{token} i was brought in to play the pessimist this morning{token}",
    },
    {
        "role": "assistant",
        "content": "Good morning. I think, as a grumpy Eastern European, I was brought in to play the pessimist this morning.",
    },
    {
        "role": "user",
        "content": "i am indeed going to talk about the spaces men create for themselves{token}",
    },
    {
        "role": "assistant",
        "content": "I am indeed going to talk about the spaces men create for themselves,",
    },
    {
        "role": "user",
        "content": "the substance of things unseen{token} cities{token} past and future{token}",
    },
    {
        "role": "assistant",
        "content": "The substance of things unseen. Cities, past and future.",
    },
    {
        "role": "user",
        "content": "moment of crisis{token} arguably the financial markets have failed us and the aid system is failing us{token} and yet i stand",
    },
    {
        "role": "assistant",
        "content": "moment of crisis. Arguably the financial markets have failed us and the aid system is failing us, and yet I stand",
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

LLM_CHAT_MESSAGES_RAW_5_EN_SAMPLE = [
    {
        "role": "system",
        "content": "Restore the punctuation to the following text without any extra explanation. ",
    },
    {
        "role": "user",
        "content": "most of the time art and science stare at each other across a gulf of mutual incomprehension there is great confusion",
    },
    {
        "role": "assistant",
        "content": "Most of the time, art and science stare at each other across a gulf of mutual incomprehension. There is great confusion",
    },
    {
        "role": "user",
        "content": "good morning i think as a grumpy eastern european i was brought in to play the pessimist this morning",
    },
    {
        "role": "assistant",
        "content": "Good morning. I think, as a grumpy Eastern European, I was brought in to play the pessimist this morning.",
    },
    {
        "role": "user",
        "content": "i am indeed going to talk about the spaces men create for themselves",
    },
    {
        "role": "assistant",
        "content": "I am indeed going to talk about the spaces men create for themselves,",
    },
    {
        "role": "user",
        "content": "the substance of things unseen cities past and future",
    },
    {
        "role": "assistant",
        "content": "The substance of things unseen. Cities, past and future.",
    },
    {
        "role": "user",
        "content": "moment of crisis arguably the financial markets have failed us and the aid system is failing us and yet i stand",
    },
    {
        "role": "assistant",
        "content": "moment of crisis. Arguably the financial markets have failed us and the aid system is failing us, and yet I stand",
    },
    {
        "role": "user",
        "content": "东方主义的影响非常广泛它不仅影响了西方对东方世界的看法还对艺术文学音乐电影等多个领域产生了重要影响",
    },
    {
        "role": "assistant",
        "content": "东方主义的影响非常广泛，它不仅影响了西方对东方世界的看法还对艺术，文学，音乐电影等多个领域产生了重要影响。",
    },
]

LLM_CHAT_MESSAGES_FIND_POSITION = [
    {
        "role": "system",
        "content": "Place {token} after each English word or Chinese character where punctuation is needed.",
    },
    {
        "role": "user",
        "content": "it involves developing skills such as listening speaking pronunciation vocabulary grammar and discourse",
    },
    {
        "role": "assistant",
        "content": "it involves developing skills such as listening{token} speaking{token} pronunciation{token} vocabulary{token} grammar{token} and discourse{token}",
    },
    {"role": "user", "content": "东方主义的影响非常广泛它不仅影响了西方对东方世界的看法还对艺术文学音乐电影等多个领域产生了重要影响"},
    {
        "role": "assistant",
        "content": "东方主义的影响非常广泛{token}它不仅影响了西方对东方世界的看法{token}还对艺术{token}文学{token}音乐电影等多个领域产生了重要影响{token}",
    },
]

LLM_CHAT_MESSAGES_REPETATION = [
    {
        "role": "system",
        "content": "Repeat the input text.",
    },
    {
        "role": "user",
        "content": "it involves developing skills such as listening speaking pronunciation vocabulary grammar and discourse",
    },
    {
        "role": "assistant",
        "content": "it involves developing skills such as listening speaking pronunciation vocabulary grammar and discourse",
    },
    {"role": "user", "content": "东方主义的影响非常广泛它不仅影响了西方对东方世界的看法还对艺术文学音乐电影等多个领域产生了重要影响"},
    {
        "role": "assistant",
        "content": "东方主义的影响非常广泛它不仅影响了西方对东方世界的看法还对艺术文学音乐电影等多个领域产生了重要影响",
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

PUNCTSPECIAL2LABLE = {PUNCT_SPECIAL_TOKEN: "PUNCT"}

LABEL2ID = {
    NORMAL_TOKEN_TAG: 0,
    "COMMA": 1,
    "PERIOD": 2,
    "QUESTIONMARK": 3,
    "EXCLAMATIONMARK": 4,
}

PUNCTSPECIAL2ID = {NORMAL_TOKEN_TAG: 0, "PUNCT": 1}

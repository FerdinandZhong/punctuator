import logging

from dbpunctuator.inference import Inference, InferenceArguments
from dbpunctuator.utils import CHINESE_PUNCS, DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP
from dbpunctuator.utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def produce_sample_text(text):
    puncs = str.maketrans("", "", CHINESE_PUNCS + "\n")
    return text.translate(puncs)


if __name__ == "__main__":
    args = InferenceArguments(
        model_name_or_path="models/chinese_punctuator",
        tokenizer_name="bert-base-chinese",
        tag2punctuator=DEFAULT_CHINESE_TAG_PUNCTUATOR_MAP,
    )

    inference = Inference(inference_args=args, verbose=False)

    test_texts_1 = [
        """
        我想指出，中方这一举措是对立方损害中国主权的正当反制，责任完全在立方。中国人民不可侮，中国国家主权和领土完整不容侵犯。中国政府维护国家主权安全和发展利益的决心坚定不移。任何挑战中国核心利益的行径都不会得逞。我们奉劝立方恪守公认的国际关系基本准则，立即纠正错误，重新回到正确轨道上来。
        """
    ]
    logger.info(
        f"testing result {inference.punctuation([produce_sample_text(text) for text in test_texts_1])}"
    )

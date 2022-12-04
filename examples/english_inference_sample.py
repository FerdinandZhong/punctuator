import logging

from dbpunctuator.inference import Inference, InferenceArguments
from dbpunctuator.utils import ALL_PUNCS, DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP
from dbpunctuator.utils.utils import register_logger

logger = logging.getLogger(__name__)
register_logger(logger)


def produce_sample_text(text, repl=None):
    puncs = dict(zip(ALL_PUNCS, [repl] * len(ALL_PUNCS)))
    return text.lower().translate(puncs)


test_texts_1 = [
    "how are you its been ten years since we met in shanghai im really happy to meet you again whats your current phone number",  # noqa: E501
    "my number is 82732212",
]

test_texts_2 = [
    "hi sir have you registered your account in our platform its having a big sale right now",
    "oh yes i have its a good website",
    "great thank you sir here is an additional promo code 5566",
]

long_test_text = [
    """
    Meanwhile, Giannis Antetokounmpo had 24 points, 13 rebounds and seven assists as the Milwaukee Bucks went wire-to-wire to capture their sixth consecutive victory with a 120-109 win over the Denver Nuggets on Friday.
    Pat Connaughton poured in 20 points and Khris Middleton added 17 for the Bucks, who are now four games above the .500 win mark.
    Jrue Holiday scored 16 points, Bobby Portis tallied 11 and Grayson Allen finished with 10.
    “I feel like it’s just comfortable,” Holiday said. “We went through some struggles and that helped us to kind of find a place.” Aaron Gordon scored 18 points and nine rebounds for Denver, who lost their sixth straight.
    Will Barton tallied 17 points and nine rebounds while Facundo Campazzo had 16 points as Denver fell behind by 20 points in the second half and shot just 17 of 47 from beyond the arc.
    """,  # noqa: E501
    """
    Opening my article let me guess it’s safe to assume that you have heard of BERT. If you haven’t, or if you’d like a refresh, I recommend reading this paper.

    In deep learning, there are currently two options for how to build language models. You can build either monolingual models or multilingual models.

    “multilingual, or not multilingual, that is the question” — as Shakespeare would have said

    Multilingual models describe machine learning models that can understand different languages. An example of a multilingual model is mBERT from Google research. This model supports and understands 104 languages. Monolingual models, as the name suggest can understand one language.

    Multilingual models are already achieving good results on certain tasks. But these models are bigger, need more data, and also more time to be trained. These properties lead to higher costs due to the larger amount of data and time resources needed.

    Due to this fact, I am going to show you how to train a monolingual non-English BERT-based multi-class text classification model. Wow, that was a long sentence!
    """,  # noqa: E501
]

if __name__ == "__main__":
    args = InferenceArguments(
        model_weight_name="Qishuai/distilbert_punctuator_en",
        tokenizer_name="Qishuai/distilbert_punctuator_en",
        tag2punctuator=DEFAULT_ENGLISH_TAG_PUNCTUATOR_MAP,
        gpu_device=1,
    )

    inference = Inference(inference_args=args, verbose=False)

    logger.info(f"testing result {inference.punctuation(test_texts_1)}")

    logger.info(f"testing result {inference.punctuation(test_texts_2)}")

    long_test_text = [produce_sample_text(text) for text in long_test_text]
    logger.info(f"produced sample text: {long_test_text}")
    logger.info(f"testing result {inference.punctuation(long_test_text)[0]}")

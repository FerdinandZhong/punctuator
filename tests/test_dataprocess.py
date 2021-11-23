import re

import pytest

from dbpunctuator.utils.constant import DEFAULT_ENGLISH_NER_MAPPING
from tests.common import cleaned_data, processed_data  # noqa: F401

punctuations = list(DEFAULT_ENGLISH_NER_MAPPING.keys())


@pytest.mark.usefixtures("cleaned_data")
def test_data_cleaning(cleaned_data):  # noqa: F811
    checking_regex = r"\([^()]*\)"
    for line in cleaned_data:
        bracketed_texts = re.findall(checking_regex, line)
        assert len(bracketed_texts) == 0


@pytest.mark.usefixtures("processed_data")
def test_training_data_generation(processed_data):  # noqa: F811
    for tokens, tags in zip(*processed_data):
        last_token_is_punct = False
        for token, tag in zip(tokens, tags):
            assert not token.isdigit()
            if last_token_is_punct:
                assert token not in punctuations
            if token in punctuations:
                assert tag != "O"
                last_token_is_punct = True

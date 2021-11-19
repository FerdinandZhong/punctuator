import pandas as pd
import pytest

from dbpunctuator.data_process import remove_brackets_text
from dbpunctuator.data_process.data_cleanning import dataframe_data_cleaning
from dbpunctuator.data_process.data_process import process_line
from dbpunctuator.utils.constant import DEFAULT_NER_MAPPING

punctuations = list(DEFAULT_NER_MAPPING.keys())


@pytest.fixture(scope="module")
def cleaned_data():
    test_df = pd.DataFrame(
        {
            "test": [
                "Good morning. How are you?(Laughter)It's been great, hasn't it?",  # noqa: E501
                "This is a test without special character.",
                "'Hello', who @ are * you (music)?",
                "i don't know what's going on...",
                "my number is 54785564.",
                """
                We shot the scene without a single rehearsal", beatty said.
                As usual the director insisted on a rehearsal, but I convinced him the best opportunity for a realistic battle would be when the two animals first met.
                You simply can't rehearsal a scene like that. Hence the cameramen were ready and the fight was a real one, unfaked...
                And claw to claw, fang to fang battle between natrual enimies of the cat family proved conclusively that the fighting prowess of the lion is superior to that of the tiger according to beatty the tiger lost the battle after a terrific struggle.
                We used a untamed tiger for the battle scene because we figured a good fight was a likely to ensue, the trainer continued.
                That tiger never before been in a cage with a lion.
                Nearly a score of movie stars watched the battle and the majority of them bet on the tiger.
                I had no idea which would win, but I figured sultan had an even chance, though lions are gang fighters and a tiger is supposed to be invinceable in a single-handed battle with almost any animal.
                My reasons for giving the lion an even chance was that I knew that when one takes a hold with his teeth it is in a vital spot, while a tiger sinks his teeth and hangs on whereever he grabs first.
                Thats exactly why tommy lost the fight. While the tiger is simply hanging on to a shoulder, the lion was manuvering into position to get his enemys throat, all the while using his blade-like claws to great advantage, from now on I'll bet on the lion.
                """,  # noqa: E501
            ]
        }
    )

    cleaned_df = dataframe_data_cleaning(
        test_df, "test", set(DEFAULT_NER_MAPPING.keys()), [], remove_brackets_text
    )

    return cleaned_df["test"].tolist()


@pytest.fixture(scope="module")
def processed_data(cleaned_data):
    all_tokens = []
    all_tags = []
    for line in cleaned_data:
        tokens, tags = process_line(line)
        all_tokens.append(tokens)
        all_tags.append(tags)
    return all_tokens, all_tags

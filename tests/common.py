import pandas as pd
import pytest

from dbpunctuator.data_process import remove_brackets_text
from dbpunctuator.data_process.data_cleanning import dataframe_data_cleaning
from dbpunctuator.data_process.data_process import process_line
from dbpunctuator.utils import DEFAULT_ENGLISH_NER_MAPPING, ALL_PUNCS

punctuations = list(DEFAULT_ENGLISH_NER_MAPPING.keys())


@pytest.fixture(scope="module")
def cleaned_data():
    test_df = pd.DataFrame(
        {
            "test": [
                "Good morning. How are you?(Laughter)It's been great, hasn't it?",  # noqa: E501
                "This is a test without special character.",
                "'Hello', who @ are * you (music)?",
                "i don't know what's going on...",
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
                """The World Health Organization (WHO) warned on Monday that the heavily mutated Omicron coronavirus variant was likely to spread internationally and poses a very high risk of infection surges that could have "severe consequences" in some places.
                The Saudi ministry urged people to complete their vaccination and ordered travellers to respect self-isolation and testing rules.
                Saudi Arabia last week halted flights from seven southern African countries, mirroring similar moves by other governments, but travel links with North Africa have remained unaffected.
                Omicron was first reported on Nov 24 in southern Africa, where infections have risen steeply. It has since spread to more than a dozen countries, many of which have imposed travel restrictions to try to seal themselves off.
                Japan on Monday joined Israel and Morocco in saying it would completely close its borders.
                """, # noqa: E501
                """
                Rust native Transformer-based models implementation. Port of Hugging Face's Transformers library, using the tch-rs crate and pre-processing from rust-tokenizers. Supports multi-threaded tokenization and GPU inference. This repository exposes the model base architecture, task-specific heads (see below) and ready-to-use pipelines. Benchmarks are available at the end of this document.
                """ # noqa: E501
            ]
        }
    )

    kept_punctuations = [ord(p) for p in set(DEFAULT_ENGLISH_NER_MAPPING.keys())]
    removed_punctuations = [p for p in ALL_PUNCS if p not in kept_punctuations]
    cleaned_df = dataframe_data_cleaning(
        test_df,
        "test",
        kept_punctuations,
        removed_punctuations,
        remove_brackets_text,
    )

    return cleaned_df["test"].tolist()


@pytest.fixture(scope="module")
def processed_data(cleaned_data):
    all_tokens = []
    all_tags = []
    for line in cleaned_data:
        tokens, tags = process_line(line, DEFAULT_ENGLISH_NER_MAPPING)
        all_tokens.append(tokens)
        all_tags.append(tags)
    return all_tokens, all_tags

import re
from string import punctuation

import pandas as pd
from tqdm import tqdm

tqdm.pandas()

default_kept_punctuations = {",", ".", "?", "!"}


def remove_brackets_text(input):
    return re.sub(r"\([^()]*\)", " ", input)

def keep_only_latin_characters(input):
    regex = re.compile('[^\u0020-\u024F]')
    return regex.sub('', input)

def dataframe_data_cleaning(
    df,
    target_col,
    kept_punctuations={},
    additional_to_remove=[],
    *special_cleaning_func
):
    """
    Clean up data in dataframe by removing all special characters except kept ones.
    """
    if special_cleaning_func:
        for func in special_cleaning_func:
            df[target_col] = df[target_col].progress_apply(lambda x: func(x))

    removed_punctuations = "".join(
        [p for p in punctuation if p not in kept_punctuations] + additional_to_remove
    )
    translator = str.maketrans("", "", removed_punctuations)
    space_translator = str.maketrans({key: " {0}".format(key) for key in kept_punctuations})

    df[target_col] = df[target_col].progress_apply(
        lambda x: x.lower().translate(translator).translate(space_translator).strip()
    )

    df.dropna(subset=[target_col])
    return df


if __name__ == "__main__":
    df = pd.DataFrame(
        {
            "test": [
                "Good morning. How are you?(Laughter)It's been great, hasn't it?"
                "This is a test without special character!",
                "'Hello' who @ are * you (music)?",
            ]
        }
    )
    print(df.head())
    df = dataframe_data_cleaning(
        df, "test", default_kept_punctuations, [], remove_brackets_text
    )
    print(df.head())
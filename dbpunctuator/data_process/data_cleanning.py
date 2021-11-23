import re

from tqdm import tqdm

tqdm.pandas()


def dataframe_data_cleaning(
    df, target_col, kept_punctuations, removed_punctuations, *special_cleaning_funcs
):
    """
    Clean up data in dataframe by removing all special characters except kept ones.
    """
    if special_cleaning_funcs:
        for func in special_cleaning_funcs:
            df[target_col] = df[target_col].progress_apply(lambda x: func(x))

    translator = str.maketrans("", "", removed_punctuations)
    space_translator = str.maketrans(
        {key: " {0} ".format(key) for key in kept_punctuations}
    )

    df[target_col] = df[target_col].progress_apply(
        lambda x: x.lower().translate(translator).translate(space_translator).strip()
    )

    df.dropna(subset=[target_col])
    return df


def cleaning_validator(text, kept_punctuations, removed_punctuations):
    regex = re.compile(f"[{removed_punctuations}]")
    checking_result = regex.search(text)
    assert (
        checking_result is None
        or text[checking_result.span()[0] : checking_result.span()[1]]
        in kept_punctuations
    ), f"data cleaning for `{text}`` doesn't pass the validation with {checking_result}"
    return True

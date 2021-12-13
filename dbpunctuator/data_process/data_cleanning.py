import logging
import re

from plane import replace
from plane.pattern import EMAIL, TELEPHONE
from tqdm import tqdm

from dbpunctuator.utils import (
    CURRENCY,
    CURRENCY_TOKEN,
    EMAIL_TOKEN,
    NUMBER,
    NUMBER_TOKEN,
    TELEPHONE_TOKEN,
    URL,
    URL_TOKEN,
)

tqdm.pandas()
logger = logging.getLogger(__name__)


def dataframe_data_cleaning(
    df, target_col, kept_punctuations, removed_punctuations, *special_cleaning_funcs
):
    """
    Clean up data in dataframe by removing all special characters except kept ones.
    """
    if special_cleaning_funcs:
        for func in special_cleaning_funcs:
            df[target_col] = df[target_col].progress_apply(lambda x: func(x))

    # replace email with <Special> and replace url with <URL>
    logger.info("replace email with <EMAIL>")
    df[target_col] = df[target_col].progress_apply(
        lambda x: replace(x, EMAIL, EMAIL_TOKEN)
    )

    logger.info("replace url with <URL>")
    df[target_col] = df[target_col].progress_apply(lambda x: replace(x, URL, URL_TOKEN))

    logger.info("replace currency with <CURRENCY>")
    df[target_col] = df[target_col].progress_apply(
        lambda x: replace(x, CURRENCY, CURRENCY_TOKEN)
    )

    logger.info("replace telephone with <TEL>")
    df[target_col] = df[target_col].progress_apply(
        lambda x: replace(x, TELEPHONE, TELEPHONE_TOKEN)
    )

    logger.info("replace number with <NUM>")
    df[target_col] = df[target_col].progress_apply(
        lambda x: replace(x, NUMBER, NUMBER_TOKEN)
    )

    translator = str.maketrans({key: None for key in removed_punctuations})
    space_translator = str.maketrans(
        {key: " {0} ".format(chr(key)) for key in kept_punctuations}
    )

    df[target_col] = df[target_col].progress_apply(
        lambda x: x.translate(translator).translate(space_translator).strip()
    )

    df.dropna(subset=[target_col])
    return df


def text_lines_cleaning(
    input_lines, kept_punctuations, removed_punctuations, *special_cleaning_funcs
):
    logger.info("clean up text file line by line.")
    logger.info("replace email with <EMAIL>")
    logger.info("replace url with <URL>")
    logger.info("replace currency with <CURRENCY>")
    logger.info("replace telephone with <TEL>")
    logger.info("replace number with <NUM>")

    pbar = tqdm(input_lines)
    for line in pbar:
        if special_cleaning_funcs:
            for func in special_cleaning_funcs:
                try:
                    line = func(line)
                except Exception as err:
                    logger.warning(f"error {str(err)} with func {func} for line {line}")
        line = replace(line, EMAIL, EMAIL_TOKEN)

        line = replace(line, URL, URL_TOKEN)

        line = replace(line, CURRENCY, CURRENCY_TOKEN)

        line = replace(line, TELEPHONE, TELEPHONE_TOKEN)

        line = replace(line, NUMBER, NUMBER_TOKEN)

        translator = str.maketrans({key: None for key in removed_punctuations})
        space_translator = str.maketrans(
            {key: " {0} ".format(chr(key)) for key in kept_punctuations}
        )

        yield line.translate(translator).translate(space_translator).strip()

    pbar.close()


def cleaning_validator(text, kept_punctuations, removed_punctuations):
    regex = re.compile(
        "[{}]".format("|".join(map(re.escape, [chr(p) for p in removed_punctuations])))
    )
    checking_result = regex.search(text)
    assert (
        checking_result is None
        or text[checking_result.span()[0] : checking_result.span()[1]]
        in kept_punctuations
    ), f"data cleaning for `{text}`` doesn't pass the validation with {checking_result}"
    return True

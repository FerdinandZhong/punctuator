import re


def remove_brackets_text(input):
    """remove brackets and text inside

    Args:
        input (string): text to apply the regex func to

    Returns:
        [type]: [description]
    """
    return re.sub(r"\([^()]*\)", " ", input)


def keep_only_latin_characters(input):
    """keep only latin characters

    Args:
        input (string): text to apply the regex func to

    Returns:
        [type]: [description]
    """
    regex = re.compile("[^\u0020-\u024F]")
    return regex.sub("", input)

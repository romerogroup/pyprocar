import re
from typing import Union, List

def remove_comment(txt: str, comment_symbol: str="#"):
    """
    Removes all comments starting with `comment_symbol` from the given text.
    
    Args:
        txt (str): The input text containing comments.
        comment_symbol (str): The symbol that marks the start of a comment (e.g., "#", "//").
    
    Returns:
        str: The text with comments removed.
    """
    escaped_symbol = re.escape(comment_symbol)
    pattern = re.compile(rf"{escaped_symbol}.*")
    return pattern.sub("", txt)
    

def bool_fortran(string: str):
    string = string.lower()
    ret = string == '.true.'
    if not ret:
        ret = not(string == '.false.')
        if ret:
            raise ValueError(f'{string} is not a fortan boolean')
    return ret
    
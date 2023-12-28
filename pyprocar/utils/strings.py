from typing import Union, List

def remove_comment(txt: str, comment_symbol: Union[str, List[str]]=["#", '!']):
    for cmnt in comment_symbol:
        txt = txt[:txt.find(cmnt)]
    return txt
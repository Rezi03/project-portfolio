# --------------------------- Packages import ------------------------------- #

import sys
import os
import pandas as pd
import re


sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

from interface.load import load_db
from tools.constants import EN_PATH

# --------------------------- DataFrame work -------------------------------- #

df = load_db(EN_PATH)

# --------------------------------------------------------------------------- #

def check_keywords_regex(input_text, keywords) -> bool:

    """
    Checks for keywords using regular expressions (case-insensitive).
    Args:
        input_text (str): the text to search for keywords
        keywords (list): a list of keywords to search for
    Returns:
        bool: True if at least one keyword is found
    """
    input_text = input_text.lower()
    for keyword in keywords:
        pattern = r'\b' + keyword.lower() + r'\b' # \b matches word boundaries
        if re.search(pattern, input_text):
            return True
    return False

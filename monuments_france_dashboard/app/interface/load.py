#                          DATABASE LOAD

# ---------------------------------------------------------------------------- #

# ------------------------- Packages import ---------------------------------- #
import sys
import os

import pandas as pd



sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

# --------------------------------------------------------------------------- #
# --------------------------- DataFrame work -------------------------------- #


def load_db(open_db:str) -> pd.DataFrame:
    """
    Load the database in a DataFrame
    """
    df = pd.read_csv(open_db, sep=';')

    return df
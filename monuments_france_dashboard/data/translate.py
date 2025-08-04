#                          DATABASE TRANSLATION 

# ---------------------------------------------------------------------------- #

# ------------------------- Packages import ---------------------------------- #
import sys
import os

import pandas as pd

sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

from tools.constants import PATH,DICT_PATH 
from app.interface.load import load_db

# --------------------------------------------------------------------------- #
# --------------------------- Translation ----------------------------------- #

try:
    df = load_db(PATH)

    # Check if the english name column exists, if not, create it

    if 'monument_name' not in df.columns:
        df['monument_name'] = ''


    # Read the dictionary from the .txt file
    monument_dict = {}
    with open(DICT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            try:
                french, english = line.strip().split(":")
                monument_dict[french.strip()] = english.strip()
            except ValueError:
                print(f"Skipping invalid line: {line.strip()}")

    # Add the english names to the DataFrame
    # if the english name is not already present
    
    for index, row in df.iterrows():
        french_name = row['monument']
        if french_name in monument_dict and not row['monument_name']:
            df.loc[index, 'monument_name'] = monument_dict[french_name]

    df.to_csv(PATH, index=False, sep=';')
    print("English names added to BDD_EN.csv")

except FileNotFoundError:
    print("The file dictionnary.txt was not found.")
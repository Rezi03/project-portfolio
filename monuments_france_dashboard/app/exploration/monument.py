# --------------------------- Database exploration --------------------------- #

# This file contains the functions that explores the database to find 
# the monument that corresponds to the user's question.

# --------------------------- Packages import -------------------------------- #

import sys
import os
import pandas as pd


sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

from cleaning.clean import clean_name
from tools.constants import EN_PATH
from interface.load import load_db

# --------------------------- DataFrame work --------------------------------- #
df = load_db(EN_PATH)

# ---------------------------------------------------------------------------- #

def extract_monument(question:str) -> pd.DataFrame:
    """
    Exrtracts the name of the monument from the question 
    and returns the corresponding row in the database.
    
    Args:
        question (str): the question asked by the user
    Returns:
        pd.DataFrame: the row in the database corresponding to the monument
    """
    question = clean_name(question)

    for monument in df['monument_name']:
        if clean_name(monument) in question:
            return df[df['monument_name'] == monument] 
    
    for monument in df['monument_name']:
        monument_keywords = clean_name(monument).split(" ") 
        if all(keyword in question for keyword in monument_keywords): 
            return df[df['monument_name'] == monument]
    
    return None 


def ask_city(monument_name:str) -> pd.DataFrame:
    """
    Asks for the city if the monument exists in multiple cities 
    and returns the corresponding row in the database.
    Args:
        monument_name (str): the name of the monument
    Returns:
        pd.DataFrame: the row in the database corresponding to the monument
    """
    print(f"The name of the monument {monument_name} is found in several cities.",
          "Here are the available cities:")
    cities = df[df['monument_name'] == monument_name]['city'].unique()
    cities_lower = [clean_name(city) for city in cities] 
    print(", ".join(cities_lower))

    while True:
        city_input = input(f"For which city would you like to obtain data for {monument_name}? ").strip().lower()
        if city_input in cities_lower:
            original_city = cities[cities_lower.index(city_input)]
            return df[(df['monument_name'] == monument_name) & (df['city'] == original_city)]
        else:
            print("Invalid city. Please try again.")

    
def display_visitation(monument:pd.DataFrame, year:int) -> None:
    """Display the number of visits for a monument and a year."""

    visits = monument[str(year)].iloc[0] 
    print(f"The number of visits for {monument['monument_name'].iloc[0]} in {year} is {visits}.")



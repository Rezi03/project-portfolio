# --------------------------- Year extraction ------------------------------- #
# ------------------------- Packages import ---------------------------------- #

import re
from tools.constants import YEARS_INT, YEAR_PATTERN
# --------------------------------------------------------------------------- #

def ask_year() -> int:
    """
    Asks the user for a valid year and returns it.
    Returns:
        int: the year chosen by the user
    """
    while True:
        year = input("\nWhich year would you like to choose ? (2018, 2019, 2020, 2021): ")
        if year.isdigit() and int(year) in YEARS_INT:
            return int(year)
        elif year.isdigit() and int(year) not in YEARS_INT:
            print("This year is not in our database.")
        else:
            print("Please enter a valid year (2018, 2019, 2020, 2021).")

# --------------------------------------------------------------------------- #


def get_year_from_question(question) -> int:
    """
    Extracts the year from the question and returns it.
    """
    year_pattern = YEAR_PATTERN # constante
    match_year = re.search(year_pattern, question)

    if match_year:
        return int(match_year.group(0))
    else:
        print("For which year would you like to know the number of visits?")
        return ask_year()
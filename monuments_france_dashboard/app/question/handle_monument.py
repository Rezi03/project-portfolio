# ---------------------------- Interaction  ---------------------------------- #

# This file conatins functions that handles questions about momnuments.


# ---------------------------------------------------------------------------- #

# ------------------------- Packages import ---------------------------------- #
import sys
import os



sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))


from interface.menu import menu
from exploration.monument import display_visitation
from question.handle_year import ask_year
# ---------------------------------------------------------------------------- #

def handle_monument_not_found() -> None:

    """ Handle the case where the monument is not found in the question. """

    print("Monument not found in the question.")
    menu()


def handle_additional_visits(monument_data) -> None:
    """ 
    Handle the case where the user wants to know the number of visits for another
    year. 
    """
    while True:
        choice = input("Would you like to know the number of visits for another year ? (yes/no/q): ").strip().lower()
        if choice == 'yes':
            year = ask_year()
            display_visitation(monument_data, year)
        elif choice == 'no':
            break
        elif choice == 'q':
            print("Thank you for using our system. See you soon!")
            sys.exit()
        else:
            print("Invalid response, please answer 'yes', 'no', or 'q'.")
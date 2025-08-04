# ---------------------------- Interaction  ---------------------------------- #

# This file contains the interaction function.


# ---------------------------------------------------------------------------- #

# ------------------------- Packages import ---------------------------------- #
import sys
import os



sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))


from exploration.monument import extract_monument,display_visitation,ask_city
from question.handle_year import get_year_from_question
from interface.log import bye 
from interface.menu import menu
from question.handle_monument import handle_monument_not_found,handle_additional_visits
from question.handle_keywords import check_keywords_regex
from analyse.analyse_data import visualise_mean_median,visualise_mean_median_city

# ---------------------------------------------------------------------------- #

def get_user_question() -> str:

    """ 
    Asks the user for a question.
    Returns:
        str: the user's question
    """

    return input("What is your question? ").strip().lower()



def handle_final_choices(monument_data) -> None:

    """ 
    Handles the choices of the user for the last request. 
    Args:
        monument_data (pd.DataFrame): the data of the monument
    """
    
    
    while True:
        print("\nWhat would you like to do now ?")
        print("Q. Quit")
        print("1. Know the number of visits for another monument")
        print(f"2. Know the city where {monument_data['monument_name'].iloc[0]} is located")

        final_choice = input("Enter the number corresponding to your choice (Q, 1 or 2): ").strip()

        if final_choice.upper() == 'Q':
            print("Thank you for using our system. See you soon!")
            break
        elif final_choice == '1':
            menu()
            ask_question() 
            break
        elif final_choice == '2':
            display_monument_location(monument_data)
            break
        else:
            print("Invalid response, please enter Q, 1 or 2.")

def ask_question(recursion_limit=10) -> None:
    """
    Manages questions with recursion, limiting depth to prevent infinite loops.
    """
    if recursion_limit <= 0:  # Add a termination condition for depth
        print("Error: Too many invalid attempts. Exiting.")
        bye()
        exit()

    question = get_user_question()

    if question.upper() == 'Q':
        bye()
        exit()  # Exit for 'Q'

    if check_keywords_regex(question, ["visualise", "visits"]):
        if check_keywords_regex(question, ["years"]):
            visualise_mean_median()
        elif check_keywords_regex(question, ["city"]):
            visualise_mean_median_city()
        return ask_question(recursion_limit - 1)  # Recursive call with decremented limit

    monument_data = extract_monument(question)

    if monument_data is None:
        handle_monument_not_found()
        return ask_question(recursion_limit - 1) 

    monument_name = monument_data['monument_name'].iloc[0]

    if len(monument_data['city'].unique()) > 1:
        monument_data = ask_city(monument_name)

    year = get_year_from_question(question)
    display_visitation(monument_data, year)
    handle_additional_visits(monument_data)
    handle_final_choices(monument_data)

def handle_subsequent_choices() -> None:

    """ Handles the choices of the user after the first question. """

    while True:
        print("\nWhat would you like to do now ?")
        print("Q. Quit")
        print("1. Know the number of visits for another monument")
        print("M. Return to menu")

        new_choice = input("Enter the number corresponding to your choice (Q, M or 1): ").strip()

        if new_choice.upper() == 'Q':
            print("Thank you for using our system. See you soon!")
            return
        elif new_choice == '1':
            ask_question() 
            return
        elif new_choice == 'M':
            menu()
            ask_question()
            return
        else:
            print("Invalid response, please enter Q, M or 1.")

def display_monument_location(monument_data) -> None:
    """Display the city where the monument is located."""

    print(f"The monument {monument_data['monument_name'].iloc[0]} is located in {monument_data['city'].iloc[0]}.")
    handle_subsequent_choices()

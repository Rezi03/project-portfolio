# ---------------------------- MAIN ------------------------------------------ #

# ---------------------------------------------------------------------------- #

# ------------------------- Packages import ---------------------------------- #

import sys
import os

sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

from app.interface.log import welcome
from app.interface.menu import menu
from app.interface.interact import ask_question

# ---------------------------------------------------------------------------- #


def main() -> None:
    """
    Launches the application.
    """
    welcome()
    menu()
    ask_question()
    

main()

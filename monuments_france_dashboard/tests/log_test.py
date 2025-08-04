import sys
import os

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from io import StringIO
from app.interface.log import welcome, bye

class TestMessages:

    def test_welcome(self):
        # Capture the output of the welcome function
        captured_output = StringIO()
        sys.stdout = captured_output  # Redirect stdout
        welcome()  # Call the function to test
        sys.stdout = sys.__stdout__  # Reset stdout redirection
        # Check the output
        assert captured_output.getvalue() == "\nWelcome to our request system for the attendance of french historical monuments\n\n", \
        "The welcome message output is incorrect"

    def test_bye(self):
        # Capture the output of the bye function
        captured_output = StringIO()
        sys.stdout = captured_output  # Redirect stdout
        bye()  # Call the function to test
        sys.stdout = sys.__stdout__  # Reset stdout redirection
        
        # Check the output
        assert captured_output.getvalue() == "\nThank you for using our inquiry system.\n\n", \
            "The bye message output is incorrect"
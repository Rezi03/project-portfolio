import pytest
from unittest.mock import patch, MagicMock
from io import StringIO
import sys
import pandas as pd
from app.interface.interact import get_user_question, handle_final_choices
from app.interface.menu import menu
from app.interface.interact import ask_question

class TestGetUserQuestion:
    def test_valid_question(self):
        # Simulate user input with valid data
        with patch('builtins.input', return_value="What is the attendance of the Eiffel Tower?"):
            result = get_user_question()
            assert result == "what is the attendance of the eiffel tower?", \
                f"Expected 'what is the attendance of the eiffel tower?', but got '{result}'"

    def test_question_with_extra_spaces(self):
        # Simulate user input with extra spaces
        with patch('builtins.input', return_value="  What is the attendance of Notre Dame?  "):
            result = get_user_question()
            assert result == "what is the attendance of notre dame?", \
                f"Expected 'what is the attendance of notre dame?', but got '{result}'"

    def test_empty_question(self):
        # Simulate user input with an empty string
        with patch('builtins.input', return_value="   "):
            result = get_user_question()
            assert result == "", \
                f"Expected an empty string, but got '{result}'"

    def test_question_with_mixed_case(self):
        # Simulate user input with mixed uppercase and lowercase letters
        with patch('builtins.input', return_value="HoW mAnY vIsItOrS weNt To ThE Arc de Triomphe?"):
            result = get_user_question()
            assert result == "how many visitors went to the arc de triomphe?", \
                f"Expected 'how many visitors went to the arc de triomphe?', but got '{result}'"

class TestHandleFinalChoices:
    def test_quit_choice(self):
        """Test the case where the user chooses to quit (Q)."""

        monument_data = pd.DataFrame({"monument_name": ["Eiffel Tower"]}) # Simulate input data
        captured_output = StringIO()  # Capture printed output

        with patch('builtins.input', return_value='Q'), patch('sys.stdout', new=captured_output):
            handle_final_choices(monument_data)

        output = captured_output.getvalue()
        assert "Thank you for using our system. See you soon!" in output, "Expected quit message to be displayed"

    def test_choice_1_another_monument(self,monkeypatch: pytest.MonkeyPatch):
    
        # Create a sample DataFrame for monument_data.
        monument_data = pd.DataFrame({"monument_name": ["Eiffel Tower"]})
        
        # Create counters to verify the calls.
        call_counts = {"menu": 0, "ask_question": 0}
        
        # Define fake replacements for menu() and ask_question().
        def fake_menu():
            call_counts["menu"] += 1
            print("Fake menu called")
            
        def fake_ask_question():
            call_counts["ask_question"] += 1
            print("Fake ask_question called")
        
        # Monkeypatch the menu and ask_question functions in the interact app.interface.interact.
        # It is crucial to use the same namespace that handle_final_choices uses.
        monkeypatch.setattr("app.interface.interact.menu",fake_menu)
        monkeypatch.setattr("app.interface.interact.ask_question",fake_ask_question)
        
        # Monkeypatch input() to simulate the user entering '1'.
        monkeypatch.setattr("builtins.input", lambda prompt="": "1")
        
        # Monkeypatch sys.stdout to capture printed output.
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)
        
        # Call the function under test.
        handle_final_choices(monument_data)
        
        # Retrieve the printed output.
        output = captured_output.getvalue()
        
        # Check that the prompt (from handle_final_choices) is present in the output.
        expected_prompt = (
            "\nWhat would you like to do now ?\n"
            "Q. Quit\n"
            "1. Know the number of visits for another monument\n"
            "2. Know the city where Eiffel Tower is located\n"
        )
        assert expected_prompt in output, f"Expected prompt not found. Output was:\n{output}"
        
        # Verify that fake_menu and fake_ask_question were each called once.
        assert call_counts["menu"] == 1, f"Expected menu() to be called once, but was called {call_counts['menu']} times."
        assert call_counts["ask_question"] == 1, f"Expected ask_question() to be called once, but was called {call_counts['ask_question']} times."

    def test_choice_2_display_location(self, monkeypatch: pytest.MonkeyPatch):
        """
        Test that when the user selects option '2', handle_final_choices:
          - prints the correct prompt,
          - calls display_monument_location() exactly once.
        """
        # Create a sample DataFrame for monument_data
        monument_data = pd.DataFrame({"monument_name": ["Eiffel Tower"]})

        # Create a counter to track calls to display_monument_location
        call_count = {"display_location": 0}

        # Define a fake display_monument_location function
        def fake_display_location(data):
            call_count["display_location"] += 1
            print(f"Fake display_monument_location called with: {data}")

        # Patch the display_monument_location function in its original app.interface.interact
        monkeypatch.setattr("app.interface.interact.display_monument_location", fake_display_location)

        # Simulate user input to select option '2'
        monkeypatch.setattr("builtins.input", lambda prompt="": "2")

        # Redirect stdout to capture printed output
        captured_output = StringIO()
        monkeypatch.setattr(sys, "stdout", captured_output)

        # Call the function under test
        handle_final_choices(monument_data)

        # Get the captured output
        output = captured_output.getvalue()

        # Expected prompt printed by handle_final_choices
        expected_prompt = (
            "\nWhat would you like to do now ?\n"
            "Q. Quit\n"
            "1. Know the number of visits for another monument\n"
            "2. Know the city where Eiffel Tower is located\n"
        )

        # Verify the prompt is in the output
        assert expected_prompt in output, f"Expected prompt not found. Output was:\n{output}"

        # Verify that display_monument_location was called exactly once
        assert call_count["display_location"] == 1, (
            f"Expected display_monument_location() to be called once, but was called {call_count['display_location']} times"
        )

    def test_invalid_choice(self):
        """Test the case where the user enters an invalid choice."""
        monument_data = pd.DataFrame({"monument_name": ["Eiffel Tower"]})
        captured_output = StringIO()

        # Mock 'input' to simulate an invalid input followed by 'Q' to exit
        with patch('builtins.input', side_effect=['invalid', 'Q']), patch('sys.stdout', new=captured_output):
            handle_final_choices(monument_data)

        output = captured_output.getvalue()
        assert "Invalid response, please enter Q, 1 or 2." in output, "Expected invalid choice message to be displayed"
        assert "Thank you for using our system. See you soon!" in output, "Expected quit message after invalid input"

class TestAskQuestion:

    def test_exit_on_q(self, monkeypatch: pytest.MonkeyPatch):
        """Test that the function exits when 'Q' is entered."""
        # Mock input to return 'Q'
        monkeypatch.setattr("builtins.input", lambda prompt="": "Q")
                
        # Mock bye and exit functions
        bye_called = False
        def mock_bye():
            nonlocal bye_called
            bye_called = True
                
        monkeypatch.setattr("app.interface.interact.bye", mock_bye)
        monkeypatch.setattr("sys.exit", lambda: None)
                
        # Catch the SystemExit exception
        with pytest.raises(SystemExit):
            ask_question()

        assert bye_called, "Expected bye() to be called when 'Q' is entered."

    def test_ask_question_visualise_years(self):
        with patch('builtins.input', side_effect=['visualise visits years', 'Q']) as mock_input, \
            patch('app.interface.interact.bye') as mock_bye, \
            patch('app.interface.interact.visualise_mean_median') as mock_visualise:
            
            with pytest.raises(SystemExit):
                ask_question(recursion_limit=2)
            mock_visualise.assert_called_once()  # Vérifie que visualise_mean_median est appelé

    def test_ask_question_visualise_city(self):
        with patch('builtins.input', side_effect=['visualise visits city', 'Q']) as mock_input, \
            patch('app.interface.interact.bye') as mock_bye, \
            patch('app.interface.interact.visualise_mean_median_city') as mock_visualise_city:
            
            with pytest.raises(SystemExit):
                ask_question(recursion_limit=2)
            mock_visualise_city.assert_called_once()  # Vérifie que visualise_mean_median_city est appelé

    def test_ask_question_monument_not_found(self):
        with patch('builtins.input', side_effect=['unknown monument', 'Q']) as mock_input, \
            patch('app.interface.interact.handle_monument_not_found') as mock_handle_not_found, \
            patch('app.interface.interact.bye') as mock_bye:
            
            with pytest.raises(SystemExit):
                ask_question(recursion_limit=2)
            mock_handle_not_found.assert_called_once()  # Vérifie que handle_monument_not_found est appelé

    def test_ask_question_with_valid_monument(self):

        monument_data = pd.DataFrame({
        'monument_name': ['monument'],
        'city': ['city']
        })

        with patch('builtins.input', side_effect=(['monument', 'Q'])), \
            patch('app.interface.interact.extract_monument', return_value=monument_data), \
            patch('app.interface.interact.get_year_from_question', return_value=2023), \
            patch('app.interface.interact.display_visitation') as mock_display, \
            patch('app.interface.interact.handle_additional_visits'), \
            patch('app.interface.interact.handle_final_choices'), \
            patch('app.interface.interact.bye'):
            
           
            ask_question(recursion_limit=2)

            mock_display.assert_called_once()  # Vérifie que display_visitation est appelé

    def test_ask_question_exceeds_recursion_limit(self):
        with patch('builtins.input', side_effect=['invalid input', 'invalid input', 'invalid input']), \
            patch('app.interface.interact.bye') as mock_bye:
            
            with pytest.raises(SystemExit):  # Vérifie que l'exception est levée après trop d'échecs
                ask_question(recursion_limit=3)



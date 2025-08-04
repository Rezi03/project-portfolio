import sys
import os
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.exploration.monument import extract_monument, ask_city,display_visitation,df


class TestExtractMonument:
    def test_find_monument_directly(self):
    # Test with the exact name of the monument in the question
        question = "What can you tell me about the Eiffel Tower?"
        result = extract_monument(question)
        assert result is not None, "Expected a result, but got None"
        assert result['monument_name'].iloc[0] == "Eiffel Tower", "Expected 'Eiffel Tower', but got something else"

    def test_find_monument_keywords(self):
        # Test with keywords associated with the monument
        question = "Tell me about the Towers of the Cathedral in Reims."
        result = extract_monument(question)
        assert result is not None, "Expected a result, but got None"
        assert result['monument_name'].iloc[0] == "Towers", "Expected 'Towers', but got something else"

    def test_no_monument_found(self):
        # Test when no monument is mentioned
        question = "What is a good restaurant in Paris?"
        result = extract_monument(question)
        assert result is None, "Expected None, but got a result"

    def test_find_monument_with_mixed_case(self):
        # Test with a variation in letter casing
        question = "How many visitors went to the Arc De Triomphe?"
        result = extract_monument(question)
        assert result is not None, "Expected a result, but got None"
        assert result['monument_name'].iloc[0] == "Arc De Triomphe", "Expected 'Arc De Triomphe', but got something else"

    def test_find_monument_with_keywords_variations(self):
        # Test with slight variations in keywords
        question = "Tell me about the site de Glanum and its history."
        result = extract_monument(question)
        assert result is not None, "Expected a result, but got None"
        assert result['monument_name'].iloc[0] == "Glanum Site", "Expected 'Glanum Site', but got something else"


class TestAskCity:
    @patch('builtins.input', side_effect=['paris'])
    @patch('builtins.print')  # Empêche l'affichage pendant les tests
    def test_valid_city(self, mock_print, mock_input):
        result = ask_city("Eiffel Tower")
        assert result is not None
        assert result['city'].iloc[0] == 'PARIS  '

    @patch('builtins.input', side_effect=['unknown city', 'paris'])
    @patch('builtins.print')
    def test_invalid_city_then_valid(self, mock_print, mock_input):
        # L'utilisateur entre une ville invalide, puis une valide
        result = ask_city("Eiffel Tower")
        assert result is not None
        assert result['city'].iloc[0] == 'PARIS  '


    @patch('builtins.input', side_effect= ['unknown city', 'paris'])
    @patch('builtins.print')  # Empêche l'affichage pendant les tests
    def test_invalid_city_followed_by_valid(self,mock_print, mock_input):
        # L'utilisateur entre une ville invalide suivie d'une valide
        result = ask_city("Eiffel Tower")
        assert result is not None
        assert result['city'].iloc[0] == 'PARIS  '

class TestDisplayVisitation:
    @patch('builtins.print')  # Empêche l'affichage pendant les tests
    def test_display_visitation(self, mock_print):
        year = 2019
        display_visitation(df[df['monument_name'] == 'Eiffel Tower'], year)
        # Vérifie que print a été appelé avec le bon message
        mock_print.assert_called_once_with("The number of visits for Eiffel Tower in 2019 is 6 140 000.")

    @patch('builtins.print')  # Empêche l'affichage pendant les tests
    def test_display_visitation_for_another_year(self, mock_print):
        year = 2020
        display_visitation(df[df['monument_name'] == 'Eiffel Tower'], year)
        # Vérifie que print a été appelé avec le bon message
        mock_print.assert_called_once_with("The number of visits for Eiffel Tower in 2020 is 1 560 000.")
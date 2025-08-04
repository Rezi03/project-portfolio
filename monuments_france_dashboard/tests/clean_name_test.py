import sys
import os
import unittest

sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.cleaning.clean import clean_name



class TestCleanName(unittest.TestCase):

    def test_clean_basic(self):
        self.assertEqual(clean_name(" Alice "), "alice")

    def test_clean_with_extra_spaces(self):
        self.assertEqual(clean_name("   Bob   "), "bob")

    def test_clean_no_spaces(self):
        self.assertEqual(clean_name("Charlie"), "charlie")

    def test_clean_uppercase(self):
        self.assertEqual(clean_name("DAVE"), "dave")

    def test_clean_mixed_case(self):
        self.assertEqual(clean_name("eLle"), "elle")

    def test_clean_empty_string(self):
        self.assertEqual(clean_name(""), "")

    def test_clean_only_spaces(self):
        self.assertEqual(clean_name("     "), "")

if __name__ == '__main__':
    unittest.main()
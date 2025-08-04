import sys
import os

sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

import pytest
import pandas as pd
import matplotlib.pyplot as plt

# import the file  
bdd = pd.read_csv("data/BDD_EN.csv", encoding="utf-8", sep=";") 

# Define the columns to analyze
colonnes_annees = ["2018", "2019", "2020", "2021"]

# Apply the conversion to numeric before the test
bdd[colonnes_annees] = bdd[colonnes_annees].apply(pd.to_numeric, errors='coerce')

mean_values = bdd[colonnes_annees].mean()

median_values = bdd[colonnes_annees].median()


###########

def test_mean_median():
    """
    Tests whether the means and medians of the 
    specified columns are calculated correctly.
    """
    
    assert isinstance(mean_values, pd.Series) 

    #check mean_values is a Series pandas.
    

    assert isinstance(median_values, pd.Series) 
    
    # check median_values is a Series pandas.
    

    assert set(mean_values.index) == set(colonnes_annees) 
    
    # Check that all expected columns are present


    assert set(median_values.index) == set(colonnes_annees)
    
    # Checks that the columns are exactly as expected
    

    assert all(isinstance(val, (int, float)) for val in mean_values)
    assert all(isinstance(val, (int, float)) for val in median_values)
    
    # Check that the values ​​are numeric (float)
    
    assert all(mean_values >= 0)
    assert all(median_values >= 0) 
    
    # Check that the mean and median are not negative if the data should be positive
    


#################

cities_visitor_data = bdd.groupby('city')[colonnes_annees].sum().sum(axis=1).sort_values(ascending=False)
"""
Aggregation of visitors by city
"""

def test_cities_visitor_data():
    """
    Tests if the aggregation of visitors by city is correct.
    """

    assert isinstance(cities_visitor_data, pd.Series)
    """
    Test if the aggregation of visitors by city is correct.
    """

    
    unique_cities = bdd['city'].dropna().unique()
    assert set(cities_visitor_data.index) <= set(unique_cities)
    """
    Check that the index of cities_visitor_data matches the unique city in the database
    """

    assert all(isinstance(val, (int, float)) for val in cities_visitor_data)
    """
    Check that all values ​​are numeric (float or int)
    """

    assert all(cities_visitor_data >= 0)
    """
    Check that the values ​​are positive or zero (number of visitors cannot be negative)
    """

    assert all(cities_visitor_data.values[:-1] >= cities_visitor_data.values[1:])
    """
    Check that the sorting is in descending order (from largest to smallest)
    """


year_visitor_data = bdd[colonnes_annees].sum()
"""
Aggregation of visitors by year
"""

def test_year_visitor_data():
    """
    Test if the aggregation of visitors by year is correct.
    """

    assert isinstance(year_visitor_data, pd.Series)
    """
    Check that the result is a pandas Series
    """

    assert set(year_visitor_data.index) == set(colonnes_annees)
    """
    Check that the index of year_visitor_data matches the specified years
    """

    assert all(isinstance(val, (int, float)) for val in year_visitor_data)
    """
    Check that all values ​​are numeric (float or int)
    """

    assert all(year_visitor_data >= 0)
    """
    This assertion checks that all values ​​in year_visitor_data are positive or zero (≥ 0).
    """




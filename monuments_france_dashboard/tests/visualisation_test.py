import pytest

import sys
import os

sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt

from app.analyse.analyse_data import *

bdd = pd.read_csv("data/BDD_EN.csv", encoding="utf-8", sep=";") 
"""
import the file  
"""

colonnes_annees = ["2018", "2019", "2020", "2021"]
"""
Define the columns to analyse
"""

bdd[colonnes_annees] = bdd[colonnes_annees].apply(pd.to_numeric, errors='coerce')
"""
Apply the conversion to numeric before the test
"""

mean_values = bdd[colonnes_annees].mean()
"""
Calcul the mean
"""

median_values = bdd[colonnes_annees].median()
"""
Calcul the median
"""

cities_visitor_data = bdd.groupby('city')[colonnes_annees].sum().sum(axis=1).sort_values(ascending=False)
"""
Aggregation of visitors by city
"""

year_visitor_data = bdd[colonnes_annees].sum()
"""
Agrégation of visiors
"""


def test_bar_chart():
    """
    Test if the bar chart is generated without error.
    """
    
    try:
        plt.figure(figsize=(15, 8))
        cities_visitor_data.plot(kind='bar', color='skyblue')
        """
        Create the graph figure
        """

        plt.title("Monument visitation by city")
        plt.xlabel("City")
        plt.ylabel("Number of visitors")
        """
        Add titles and labels
        """

        plt.xticks(rotation=45, ha='right')
        """
        Rotate X-axis labels
        """

        plt.tight_layout()
        """
        Ajust the layout
        """

        plt.close()  
        """
        Draw the graph
        """

    except Exception as e:
        pytest.fail
        """
        The bar chart generated an error: {e}
        """

if __name__ == "__main__":
    pytest.main(["-v"])

"""
Run the test if this script is launched directly
"""

################

def test_line_chart():
    """
    Test if the bar chart is generated without error.
    """

    try:
        plt.figure(figsize=(15, 8))
        year_visitor_data.plot(kind='line', color='orange', marker='o')
        """
        Create the graph figure
        """

        plt.title("Monument visitation by year")
        plt.xlabel("Year")
        plt.ylabel("Number of visitors")
        """
        Add titles and labels
        """

        plt.grid(True)
        plt.tight_layout()
        """
        Add a grid and adjust the layout
        """

        plt.close()
        """
        Draw the graph
        """

    except Exception as e:
        pytest.fail
        """
        The bar chart generated an error: {e}
        """

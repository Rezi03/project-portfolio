# ---------------------------- DATA ANALYSIS --------------------------------- #

# ---------------------------------------------------------------------------- #

# ------------------------- Packages import ---------------------------------- #


import sys
import os

sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import matplotlib.pyplot as plt

from typing import Tuple
from interface.load import load_db
from tools.constants import PATH, YEARS

# ---------------------------------------------------------------------------- #

# --------------------------- DataFrame work --------------------------------- #

bdd = load_db(PATH)

# Convert columns to numeric (in case they are strings or other types)

bdd[YEARS] = bdd[YEARS].apply(pd.to_numeric, errors='coerce')


# ---------------------------------------------------------------------------- #

# --------------------------- DataFrame work --------------------------------- #


def calculate_mean_median() -> Tuple[pd.Series, pd.Series]:
    """
    Display the mean and median of the number of visitors per year.

    Returns:
        Tuple[pd.Series, pd.Series]: mean and median values
    """
    mean_values = bdd[YEARS].mean()
    median_values = bdd[YEARS].median()

    return mean_values, median_values



def visualise_mean_median() -> None:
    """
    Visualize the mean and median of the number of visitors per year.
    """
    mean_values, median_values = calculate_mean_median()

    # Visualization with a line chart
    plt.figure(figsize=(15, 8))
    mean_values.plot(kind='line', color='orange', marker='o', label='Mean')
    median_values.plot(kind='line', color='blue', marker='x', label='Median')
    plt.title("Mean and Median Monument Visitation by Year")
    plt.xlabel("Year")  
    plt.ylabel("Number of Visitors")
    plt.xticks(range(len(mean_values)), mean_values.index) #Clear x-axis labels
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def visualise_mean_median_bar() -> None:
    """
    Visualize the mean and median of the number of visitors per year.
    """

    mean_values, median_values = calculate_mean_median()
    
    # Visualization with a bar chart
    plt.figure(figsize=(15, 8))  # Set the figure size
    mean_values.plot(kind='bar', color='orange', label='Mean')
    median_values.plot(kind='bar', color='blue', label='Median')
    plt.title("Mean and median of monument visitation by year")
    plt.xlabel("Year")
    plt.ylabel("Number of visitors")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()  # Adjust the margins for better readability
    plt.show()


def visualise_mean_median_city() -> None:
    """
    Visualize the mean and median of the number of visitors per year by city.
    """

    # Aggregate data by city, summing the visitors for each year
    cities_visitor_data = bdd.groupby('city')[YEARS].sum().sum(axis=1).sort_values(ascending=False)

    # Visitors frequency by city

    # Visualization with a bar chart
    plt.figure(figsize=(15, 8))  # Increase the figure width
    cities_visitor_data.plot(kind='bar', color='skyblue')
    plt.title("Monument visitation by city")
    plt.xlabel("City")
    plt.ylabel("Number of visitors")
    plt.xticks(rotation=45, ha='right')  # Rotate labels by 45° and align them to the right
    plt.tight_layout()  # Automatically adjust the margins for better readability
    plt.show()

def visualise_by_region() -> None:
    """
    Visualize the number of visitors by region.
    """

    # Aggregate the data by region and sum the visitors for each year
    region_visitor_data = bdd.groupby('region')[YEARS].sum()

    # Plot the data
    plt.figure(figsize=(12, 6))
    region_visitor_data.plot(kind='bar', stacked=True)
    plt.title("Visitors by Region (2018-2021)")
    plt.xlabel("Region")
    plt.ylabel("Number of visitors")
    plt.tight_layout()
    plt.show()

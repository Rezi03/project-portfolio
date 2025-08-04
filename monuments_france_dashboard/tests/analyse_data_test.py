import sys
import os

sys.path.insert(0,os.path.abspath(
                            os.path.join(os.path.dirname(__file__), '..')))

import pytest
from app.analyse.analyse_data import visualise_mean_median, visualise_mean_median_bar, visualise_mean_median_city, calculate_mean_median

def test_calculate_mean_median():
    mean_values, median_values = calculate_mean_median()
    assert mean_values is not None, "The mean should not be None."
    assert median_values is not None, "The median should not be None."
    assert len(mean_values) > 0, "The mean should contain values."
    assert len(median_values) > 0, "The median should contain values."

def test_visualise_mean_median():
    try:
        visualise_mean_median()
    except Exception:
        pytest.fail("The function visualise_mean_median failed.")

def test_visualise_mean_median_bar():
    try:
        visualise_mean_median_bar()
    except Exception:
        pytest.fail("The function visualise_mean_median_bar failed.")

def test_visualise_mean_median_city():
    try:
        visualise_mean_median_city()
    except Exception:
        pytest.fail("The function visualise_mean_median_city failed.")
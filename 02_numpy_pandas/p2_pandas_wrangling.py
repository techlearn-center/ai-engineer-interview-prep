"""
Problem 2: Pandas Data Wrangling
=================================
Difficulty: Easy -> Medium

Pandas is essential for data manipulation in ML pipelines.
Interviewers love asking these.

Run tests:
    pytest 02_numpy_pandas/tests/test_p2_pandas_wrangling.py -v
"""
import pandas as pd
import numpy as np


def create_sample_data() -> pd.DataFrame:
    """Helper: creates sample sales data for the problems below."""
    np.random.seed(42)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=100, freq="D"),
        "product": np.random.choice(["A", "B", "C"], 100),
        "region": np.random.choice(["North", "South", "East", "West"], 100),
        "sales": np.random.randint(10, 500, 100),
        "quantity": np.random.randint(1, 50, 100),
    })


def top_products_by_region(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each region, find the product with the highest total sales.
    Return a DataFrame with columns: region, product, total_sales
    Sorted by total_sales descending.

    Hint: groupby, then idxmax or similar
    """
    # YOUR CODE HERE
    pass


def rolling_average_sales(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Add a column 'rolling_avg' that contains the 7-day rolling average of sales.
    Sort by date first. The rolling average should have min_periods=1.
    Return the DataFrame with the new column added.
    """
    # YOUR CODE HERE
    pass


def pivot_sales_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a pivot table:
        - rows: region
        - columns: product
        - values: total sales (sum)
    Fill NaN with 0.
    """
    # YOUR CODE HERE
    pass


def clean_messy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame that may have:
        - Duplicate rows
        - Missing values in 'sales' column
        - Negative values in 'quantity' column

    Clean it by:
        1. Removing exact duplicate rows
        2. Filling missing 'sales' with the median sales value
        3. Replacing negative 'quantity' with 0
        4. Resetting the index

    Return the cleaned DataFrame.
    """
    # YOUR CODE HERE
    pass

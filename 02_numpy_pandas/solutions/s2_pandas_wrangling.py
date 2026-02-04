"""
SOLUTIONS - Pandas Data Wrangling
===================================
Try to solve the problems yourself first!
"""
import pandas as pd
import numpy as np


def create_sample_data() -> pd.DataFrame:
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
    Key insight: groupby + sum, then for each region pick the max.
    idx = grouped by region, get the idxmax of total_sales.
    """
    # Group by region AND product, sum sales
    grouped = df.groupby(["region", "product"])["sales"].sum().reset_index()
    grouped.columns = ["region", "product", "total_sales"]

    # For each region, keep the product with highest total_sales
    idx = grouped.groupby("region")["total_sales"].idxmax()
    result = grouped.loc[idx].reset_index(drop=True)

    # Sort by total_sales descending
    result = result.sort_values("total_sales", ascending=False).reset_index(drop=True)
    return result


def rolling_average_sales(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Key insight: Sort by date first, then use .rolling() on the sales column.
    min_periods=1 ensures no NaN for the first few rows.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["rolling_avg"] = df["sales"].rolling(window=window, min_periods=1).mean()
    return df


def pivot_sales_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Key insight: pd.pivot_table does exactly this.
    aggfunc='sum' aggregates duplicate entries.
    """
    return pd.pivot_table(
        df,
        values="sales",
        index="region",
        columns="product",
        aggfunc="sum",
        fill_value=0,
    )


def clean_messy_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Key insight: Chain operations in order.
    Be careful: compute median BEFORE filling, using existing values.
    """
    df = df.copy()

    # 1. Remove exact duplicates
    df = df.drop_duplicates()

    # 2. Fill missing sales with median
    if "sales" in df.columns:
        median_sales = df["sales"].median()
        df["sales"] = df["sales"].fillna(median_sales)

    # 3. Replace negative quantity with 0
    if "quantity" in df.columns:
        df["quantity"] = df["quantity"].clip(lower=0)

    # 4. Reset index
    df = df.reset_index(drop=True)

    return df

import sys
import os
import pandas as pd
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p2_pandas_wrangling import (
    create_sample_data,
    top_products_by_region,
    rolling_average_sales,
    pivot_sales_report,
    clean_messy_data,
)


class TestTopProductsByRegion:
    def test_correct_columns(self):
        df = create_sample_data()
        result = top_products_by_region(df)
        assert list(result.columns) == ["region", "product", "total_sales"]

    def test_one_per_region(self):
        df = create_sample_data()
        result = top_products_by_region(df)
        assert len(result) == df["region"].nunique()

    def test_sorted_descending(self):
        df = create_sample_data()
        result = top_products_by_region(df)
        assert list(result["total_sales"]) == sorted(result["total_sales"], reverse=True)


class TestRollingAverageSales:
    def test_has_column(self):
        df = create_sample_data()
        result = rolling_average_sales(df, window=7)
        assert "rolling_avg" in result.columns

    def test_no_nulls(self):
        df = create_sample_data()
        result = rolling_average_sales(df, window=7)
        assert result["rolling_avg"].isna().sum() == 0

    def test_first_value_equals_first_sale(self):
        df = create_sample_data().sort_values("date")
        result = rolling_average_sales(df, window=7)
        result = result.sort_values("date").reset_index(drop=True)
        assert result.loc[0, "rolling_avg"] == result.loc[0, "sales"]


class TestPivotSalesReport:
    def test_shape(self):
        df = create_sample_data()
        result = pivot_sales_report(df)
        assert result.shape[0] == df["region"].nunique()
        assert result.shape[1] == df["product"].nunique()

    def test_no_nans(self):
        df = create_sample_data()
        result = pivot_sales_report(df)
        assert result.isna().sum().sum() == 0


class TestCleanMessyData:
    def test_removes_duplicates(self):
        df = pd.DataFrame({
            "sales": [100, 100, 200],
            "quantity": [5, 5, 10],
        })
        result = clean_messy_data(df)
        assert len(result) == 2

    def test_fills_missing_sales(self):
        df = pd.DataFrame({
            "sales": [100, np.nan, 300],
            "quantity": [5, 10, 15],
        })
        result = clean_messy_data(df)
        assert result["sales"].isna().sum() == 0
        assert result.loc[1, "sales"] == 200.0  # median of 100, 300

    def test_fixes_negative_quantity(self):
        df = pd.DataFrame({
            "sales": [100, 200],
            "quantity": [-5, 10],
        })
        result = clean_messy_data(df)
        assert (result["quantity"] >= 0).all()

    def test_resets_index(self):
        df = pd.DataFrame({
            "sales": [100, 200, 300],
            "quantity": [5, 10, 15],
        })
        result = clean_messy_data(df)
        assert list(result.index) == list(range(len(result)))

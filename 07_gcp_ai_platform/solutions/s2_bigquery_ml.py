"""
SOLUTIONS - BigQuery & BigQuery ML
=====================================
Try to solve the problems yourself first!
"""


def get_feature_stats(client, table_id, feature_col, group_col):
    """
    Key insight: Standard GROUP BY with aggregate functions.
    This is bread-and-butter data exploration SQL.
    """
    sql = f"""
        SELECT {group_col}, COUNT(*) AS count, AVG({feature_col}) AS avg_value
        FROM `{table_id}`
        GROUP BY {group_col}
        ORDER BY count DESC
    """
    result = client.query(sql)
    return result.to_dataframe()


def find_outliers(client, table_id, value_col, threshold):
    """Simple WHERE filter with ORDER BY."""
    sql = f"""
        SELECT *
        FROM `{table_id}`
        WHERE {value_col} > {threshold}
        ORDER BY {value_col} DESC
    """
    result = client.query(sql)
    return result.to_dataframe()


def get_top_n(client, table_id, sort_col, n=10):
    """ORDER BY + LIMIT is the pattern for top-N queries."""
    sql = f"""
        SELECT *
        FROM `{table_id}`
        ORDER BY {sort_col} DESC
        LIMIT {n}
    """
    result = client.query(sql)
    return result.to_dataframe()


def build_training_query(table_id, feature_cols, label_col,
                         where_clause=None, sample_pct=None):
    """
    Key insight: Building SQL dynamically is common in ML pipelines.
    You construct the query based on configuration, then execute it.
    """
    columns = ", ".join(feature_cols + [label_col])
    sql = f"SELECT {columns}\nFROM `{table_id}`"

    if sample_pct is not None:
        sql += f"\nTABLESAMPLE SYSTEM ({sample_pct} PERCENT)"

    if where_clause:
        sql += f"\nWHERE {where_clause}"

    return sql


def build_bqml_create_model(model_id, model_type, table_id, label_col,
                            feature_cols=None, options=None):
    """
    Key insight: BigQuery ML lets you train models WITH SQL.
    This is powerful for quick experiments without leaving BigQuery.

    In an interview, mention that BQML supports:
    - LOGISTIC_REG, LINEAR_REG (classic ML)
    - BOOSTED_TREE_CLASSIFIER/REGRESSOR (gradient boosting)
    - KMEANS (clustering)
    - DNN_CLASSIFIER/REGRESSOR (neural networks)
    - TRANSFORM for feature preprocessing
    """
    # Build OPTIONS clause
    options_parts = [
        f"model_type='{model_type}'",
        f"input_label_cols=['{label_col}']",
    ]
    if options:
        for key, value in options.items():
            if isinstance(value, str):
                options_parts.append(f"{key}='{value}'")
            else:
                options_parts.append(f"{key}={value}")

    options_str = ",\n    ".join(options_parts)

    # Build SELECT clause
    if feature_cols:
        columns = ", ".join(feature_cols + [label_col])
    else:
        columns = "*"

    sql = f"""CREATE OR REPLACE MODEL `{model_id}`
OPTIONS(
    {options_str}
) AS
SELECT {columns}
FROM `{table_id}`"""

    return sql

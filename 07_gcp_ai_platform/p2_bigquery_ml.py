"""
Problem 2: BigQuery Operations & BigQuery ML
==============================================
Difficulty: Medium

BigQuery is where your data lives at scale.
This tests your SQL + Python SDK skills using mocks.

Run tests:
    pytest 07_gcp_ai_platform/tests/test_p2_bigquery_ml.py -v
"""
import re


class MockQueryResult:
    """Simulates BigQuery query results."""
    def __init__(self, rows: list[dict], schema: list[str]):
        self.rows = rows
        self.schema = schema
        self.total_rows = len(rows)

    def to_dataframe(self):
        """Convert to a list of dicts (simulating pandas DataFrame)."""
        return self.rows


class MockBigQueryClient:
    """Simulates google.cloud.bigquery.Client."""
    def __init__(self):
        self._tables: dict[str, list[dict]] = {}

    def load_table(self, table_id: str, rows: list[dict]):
        """Helper to pre-load data for testing."""
        self._tables[table_id] = rows

    def query(self, sql: str) -> MockQueryResult:
        """
        Simple SQL executor that supports:
        - SELECT columns FROM table
        - WHERE simple conditions
        - GROUP BY with COUNT, SUM, AVG
        - ORDER BY column ASC/DESC
        - LIMIT n
        This is simplified - real BigQuery handles much more.
        """
        # This is pre-implemented for you - just focus on the functions below
        return self._execute_simple_sql(sql)

    def _execute_simple_sql(self, sql: str) -> MockQueryResult:
        """Minimal SQL parser for testing purposes."""
        sql = sql.strip().rstrip(";")
        # Find table name
        from_match = re.search(r'FROM\s+`?(\S+?)`?\s*', sql, re.IGNORECASE)
        if not from_match:
            return MockQueryResult([], [])
        table_id = from_match.group(1).strip("`")
        rows = self._tables.get(table_id, [])
        if not rows:
            return MockQueryResult([], [])

        # WHERE clause
        where_match = re.search(r'WHERE\s+(.+?)(?:GROUP|ORDER|LIMIT|$)', sql, re.IGNORECASE)
        if where_match:
            condition = where_match.group(1).strip()
            rows = self._apply_where(rows, condition)

        # GROUP BY
        group_match = re.search(r'GROUP\s+BY\s+(\w+)', sql, re.IGNORECASE)
        select_match = re.search(r'SELECT\s+(.+?)\s+FROM', sql, re.IGNORECASE)
        if group_match and select_match:
            group_col = group_match.group(1)
            select_clause = select_match.group(1)
            rows = self._apply_group_by(rows, group_col, select_clause)

        # ORDER BY
        order_match = re.search(r'ORDER\s+BY\s+(\w+)(?:\s+(ASC|DESC))?', sql, re.IGNORECASE)
        if order_match:
            col = order_match.group(1)
            desc = order_match.group(2) and order_match.group(2).upper() == "DESC"
            rows = sorted(rows, key=lambda r: r.get(col, 0), reverse=desc)

        # LIMIT
        limit_match = re.search(r'LIMIT\s+(\d+)', sql, re.IGNORECASE)
        if limit_match:
            rows = rows[:int(limit_match.group(1))]

        schema = list(rows[0].keys()) if rows else []
        return MockQueryResult(rows, schema)

    def _apply_where(self, rows, condition):
        """Very simple WHERE: supports col = val, col > val, col >= val."""
        for op in [">=", "<=", "!=", ">", "<", "="]:
            if op in condition:
                parts = condition.split(op, 1)
                col = parts[0].strip()
                val = parts[1].strip().strip("'\"")
                try:
                    val = float(val)
                except ValueError:
                    pass
                filtered = []
                for r in rows:
                    rv = r.get(col)
                    if rv is None:
                        continue
                    if op == "=" and rv == val:
                        filtered.append(r)
                    elif op == ">" and rv > val:
                        filtered.append(r)
                    elif op == ">=" and rv >= val:
                        filtered.append(r)
                    elif op == "<" and rv < val:
                        filtered.append(r)
                    elif op == "<=" and rv <= val:
                        filtered.append(r)
                    elif op == "!=" and rv != val:
                        filtered.append(r)
                return filtered
        return rows

    def _apply_group_by(self, rows, group_col, select_clause):
        """Simple GROUP BY with COUNT, SUM, AVG aggregates."""
        groups = {}
        for r in rows:
            key = r.get(group_col)
            if key not in groups:
                groups[key] = []
            groups[key].append(r)

        result = []
        agg_patterns = re.findall(r'(COUNT|SUM|AVG)\((\*|\w+)\)\s+(?:AS\s+)?(\w+)', select_clause, re.IGNORECASE)

        for key, group_rows in groups.items():
            row = {group_col: key}
            for func, col, alias in agg_patterns:
                func = func.upper()
                if func == "COUNT":
                    row[alias] = len(group_rows)
                elif func == "SUM":
                    row[alias] = sum(r.get(col, 0) for r in group_rows)
                elif func == "AVG":
                    vals = [r.get(col, 0) for r in group_rows]
                    row[alias] = sum(vals) / len(vals) if vals else 0
            result.append(row)
        return result


# ============================================================
# YOUR TASKS: Write functions that build and execute SQL queries.
# This tests both your SQL knowledge and Python SDK usage.
# ============================================================


def get_feature_stats(client: MockBigQueryClient, table_id: str,
                      feature_col: str, group_col: str) -> list[dict]:
    """
    Write a SQL query to compute statistics grouped by a column.

    The query should:
        - SELECT group_col, COUNT(*) AS count, AVG(feature_col) AS avg_value
        - FROM the given table
        - GROUP BY group_col
        - ORDER BY count DESC

    Execute the query and return the results as a list of dicts.

    Example:
        get_feature_stats(client, "project.dataset.users", "age", "country")
        -> [{"country": "US", "count": 500, "avg_value": 34.2}, ...]
    """
    # YOUR CODE HERE
    pass


def find_outliers(client: MockBigQueryClient, table_id: str,
                  value_col: str, threshold: float) -> list[dict]:
    """
    Write a SQL query to find rows where value_col > threshold.

    The query should:
        - SELECT * FROM the table
        - WHERE value_col > threshold
        - ORDER BY value_col DESC

    Return results as a list of dicts.

    Example:
        find_outliers(client, "project.data.metrics", "latency", 1000)
        -> rows where latency > 1000
    """
    # YOUR CODE HERE
    pass


def get_top_n(client: MockBigQueryClient, table_id: str,
              sort_col: str, n: int = 10) -> list[dict]:
    """
    Write a SQL query to get the top N rows sorted by sort_col descending.

    The query should:
        - SELECT * FROM the table
        - ORDER BY sort_col DESC
        - LIMIT n

    Return results as a list of dicts.
    """
    # YOUR CODE HERE
    pass


def build_training_query(table_id: str, feature_cols: list[str],
                         label_col: str, where_clause: str = None,
                         sample_pct: float = None) -> str:
    """
    Build a BigQuery SQL query for extracting ML training data.

    This is a QUERY BUILDER - it returns the SQL string, doesn't execute it.

    The query should:
        - SELECT the feature_cols and label_col
        - FROM the table_id (wrap in backticks)
        - Optionally add WHERE clause if provided
        - Add TABLESAMPLE SYSTEM if sample_pct provided (e.g., "TABLESAMPLE SYSTEM (10 PERCENT)")

    Example:
        build_training_query(
            "project.dataset.features",
            ["age", "income", "tenure"],
            "churned",
            where_clause="age > 18",
            sample_pct=10,
        )
        -> '''SELECT age, income, tenure, churned
              FROM `project.dataset.features`
              TABLESAMPLE SYSTEM (10 PERCENT)
              WHERE age > 18'''
    """
    # YOUR CODE HERE
    pass


def build_bqml_create_model(model_id: str, model_type: str,
                            table_id: str, label_col: str,
                            feature_cols: list[str] = None,
                            options: dict = None) -> str:
    """
    Build a BigQuery ML CREATE MODEL SQL statement.

    This is what BigQuery ML looks like - you train models with SQL!

    The SQL should follow this pattern:
        CREATE OR REPLACE MODEL `{model_id}`
        OPTIONS(
            model_type='{model_type}',
            input_label_cols=['{label_col}']
            {, additional options}
        ) AS
        SELECT {columns}
        FROM `{table_id}`

    If feature_cols is None, select all: SELECT *
    If options is provided, add each as key=value in OPTIONS.

    Valid model_types: LOGISTIC_REG, LINEAR_REG, KMEANS,
                       BOOSTED_TREE_CLASSIFIER, BOOSTED_TREE_REGRESSOR, DNN_CLASSIFIER

    Example:
        build_bqml_create_model(
            "project.dataset.my_model",
            "LOGISTIC_REG",
            "project.dataset.training_data",
            "churned",
            feature_cols=["age", "income"],
            options={"max_iterations": 20},
        )
    """
    # YOUR CODE HERE
    pass

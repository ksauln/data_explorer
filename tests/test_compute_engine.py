from __future__ import annotations

import unittest

import pandas as pd

from src.compute_engine import (
    extract_rationale,
    extract_sql,
    run_sql_query,
    validate_sql,
)


class ComputeEngineTests(unittest.TestCase):
    def test_extract_sql_from_fenced_block(self) -> None:
        response = "Rationale: test\n```sql\nSELECT segment, SUM(revenue) AS total FROM data GROUP BY 1;\n```"
        sql = extract_sql(response)
        self.assertEqual(sql, "SELECT segment, SUM(revenue) AS total FROM data GROUP BY 1;")

    def test_extract_rationale(self) -> None:
        rationale = extract_rationale("Rationale: grouped by region\nSQL: SELECT 1;")
        self.assertEqual(rationale, "grouped by region")

    def test_validate_sql_blocks_unsafe_statement(self) -> None:
        with self.assertRaises(ValueError):
            validate_sql("DROP TABLE data;")

    def test_run_sql_query_applies_limit(self) -> None:
        frame = pd.DataFrame({"segment": ["A", "B", "A"], "revenue": [10, 20, 30]})
        result = run_sql_query(
            frame,
            "SELECT segment, revenue FROM data ORDER BY revenue DESC",
            result_limit=2,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["revenue"], 30)

    def test_validate_sql_accepts_with(self) -> None:
        validate_sql("WITH ranked AS (SELECT * FROM data) SELECT * FROM ranked")

    def test_run_sql_query_supports_additional_tables_for_joins(self) -> None:
        fact = pd.DataFrame(
            {
                "customer_id": ["C001", "C002", "C001"],
                "net_sales": [100.0, 200.0, 50.0],
            }
        )
        customers = pd.DataFrame(
            {
                "customer_id": ["C001", "C002"],
                "segment": ["Enterprise", "SMB"],
            }
        )
        result = run_sql_query(
            fact,
            """
            SELECT c.segment, SUM(f.net_sales) AS total_sales
            FROM data f
            JOIN customers c ON f.customer_id = c.customer_id
            GROUP BY 1
            ORDER BY total_sales DESC
            """,
            additional_tables={"customers": customers},
            result_limit=10,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result.iloc[0]["segment"], "SMB")
        self.assertEqual(result.iloc[0]["total_sales"], 200.0)

    def test_run_sql_query_handles_date_trunc_on_string_date(self) -> None:
        frame = pd.DataFrame(
            {
                "sale_date": ["2025-01-01", "2025-01-03", "2025-02-01"],
                "quantity": [2, 3, 1],
            }
        )
        result = run_sql_query(
            frame,
            """
            SELECT date_trunc('month', sale_date) AS month, SUM(quantity) AS total_quantity
            FROM data
            GROUP BY 1
            ORDER BY 1
            """,
            result_limit=10,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(int(result.iloc[0]["total_quantity"]), 5)


if __name__ == "__main__":
    unittest.main()
